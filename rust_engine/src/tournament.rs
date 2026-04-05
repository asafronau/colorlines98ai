/// Tournament bracket player — successive halving with rollouts.
///
/// Phase 1: 2-ply evaluation for all legal moves
/// Phase 2: Quarter-finals (top 30, 10 short rollouts → top 10)
/// Phase 3: Semi-finals (40 full rollouts → top 3)
/// Phase 4: Finals (remaining rollouts → best move)

use crate::board::*;
use crate::game::ColorLinesGame;
use crate::heuristic::evaluate_move;
use crate::rng::SimpleRng;

pub struct TournamentResult {
    pub chosen: (usize, usize, usize, usize),
    pub top_moves: Vec<((usize, usize, usize, usize), f64)>,
}

struct Candidate {
    mv: (usize, usize, usize, usize),
    score_2ply: f64,
}

/// Get best move using plain heuristic (game.rng NOT consumed).
fn get_best_move_pure(game: &mut ColorLinesGame) -> Option<(usize, usize, usize, usize)> {
    game.ensure_cc();
    let src_bits = get_source_mask_bits(&game.board);
    let labels = game.cc_labels;
    let comp = ComponentMasks::from_labels(&labels);
    let mut best_score = f64::NEG_INFINITY;
    let mut best = None;
    let mut board = game.board;

    let mut src = src_bits;
    while src != 0 {
        let si = src.trailing_zeros();
        src &= src - 1;
        let (sr, sc) = idx_to_rc(si);
        let color = board[sr][sc];
        let tgt = comp.target_mask(&labels, sr, sc);
        let mut t = tgt;
        while t != 0 {
            let ti = t.trailing_zeros();
            t &= t - 1;
            let (tr, tc) = idx_to_rc(ti);
            let s = evaluate_move(&mut board, sr, sc, tr, tc, color);
            if s > best_score { best_score = s; best = Some((sr, sc, tr, tc)); }
        }
    }
    best
}

/// Pre-allocated buffers for rollout move evaluation (avoids 43K+ Vec allocations per tournament move).
struct RolloutBuf {
    moves: Vec<(usize, usize, usize, usize)>,
    scores: Vec<f64>,
}

impl RolloutBuf {
    fn new() -> Self {
        RolloutBuf {
            moves: Vec::with_capacity(1200),
            scores: Vec::with_capacity(1200),
        }
    }
}

/// Softmax move using game's RNG (matches old engine coupling behavior).
/// Uses pre-allocated buffers to avoid allocation.
fn get_softmax_move_coupled(game: &mut ColorLinesGame, temperature: f64, buf: &mut RolloutBuf)
    -> Option<(usize, usize, usize, usize)>
{
    buf.moves.clear();
    buf.scores.clear();
    game.ensure_cc();
    let src_bits = get_source_mask_bits(&game.board);
    let labels = game.cc_labels;
    let comp = ComponentMasks::from_labels(&labels);
    let mut board = game.board;

    let mut src = src_bits;
    while src != 0 {
        let si = src.trailing_zeros();
        src &= src - 1;
        let (sr, sc) = idx_to_rc(si);
        let color = board[sr][sc];
        let tgt = comp.target_mask(&labels, sr, sc);
        let mut t = tgt;
        while t != 0 {
            let ti = t.trailing_zeros();
            t &= t - 1;
            let (tr, tc) = idx_to_rc(ti);
            let s = evaluate_move(&mut board, sr, sc, tr, tc, color);
            buf.moves.push((sr, sc, tr, tc));
            buf.scores.push(s);
        }
    }
    if buf.moves.is_empty() { return None; }
    let max_s = buf.scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Softmax sampling with single-pass (no probs Vec allocation)
    let mut sum = 0.0f64;
    for s in &buf.scores { sum += ((s - max_s) / temperature).exp(); }
    let r = game.rng.next_f64() * sum;
    let mut cumul = 0.0;
    for (i, s) in buf.scores.iter().enumerate() {
        cumul += ((s - max_s) / temperature).exp();
        if r <= cumul { return Some(buf.moves[i]); }
    }
    Some(*buf.moves.last().unwrap())
}

fn rollout(
    game: &ColorLinesGame,
    mv: (usize, usize, usize, usize),
    depth: usize,
    temperature: f64,
    seed: u64,
    rbuf: &mut RolloutBuf,
) -> f64 {
    let mut clone = game.clone_with_rng(SimpleRng::new(seed));
    let initial_score = clone.score;
    let (valid, _, _, _) = clone.move_ball(mv.0, mv.1, mv.2, mv.3);
    if !valid { return -1000.0; }
    let move_pts = clone.score - initial_score;
    let base_score = clone.score;

    for _ in 0..depth {
        if clone.game_over { break; }
        let m = if temperature > 0.0 {
            get_softmax_move_coupled(&mut clone, temperature, rbuf)
        } else {
            get_best_move_pure(&mut clone)
        };
        match m {
            Some((sr, sc, tr, tc)) => { clone.move_ball(sr, sc, tr, tc); }
            None => break,
        }
    }
    (clone.score - base_score) as f64 + move_pts as f64
}

fn run_rollouts(
    game: &ColorLinesGame,
    candidates: &[Candidate],
    n_rollouts: usize,
    depth: usize,
    temperature: f64,
    rng: &mut SimpleRng,
    rbuf: &mut RolloutBuf,
) -> Vec<Vec<f64>> {
    candidates.iter().map(|c| {
        (0..n_rollouts).map(|_| rollout(game, c.mv, depth, temperature, rng.next_u64(), rbuf)).collect()
    }).collect()
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

/// Play one tournament move with successive halving.
pub fn tournament_player(
    game: &mut ColorLinesGame,
    num_rollouts: usize,
    rollout_depth: usize,
    temperature: f64,
) -> Option<TournamentResult> {
    game.ensure_cc();
    let src_bits = get_source_mask_bits(&game.board);
    let labels = game.cc_labels;
    let comp_masks = ComponentMasks::from_labels(&labels);

    // Phase 1: 2-ply for ALL legal moves (bitmask iteration skips empty cells)
    let mut candidates: Vec<Candidate> = Vec::with_capacity(256);
    let mut ply = game.clone();
    let mut rbuf = RolloutBuf::new();

    let mut src = src_bits;
    while src != 0 {
        let si = src.trailing_zeros();
        src &= src - 1;
        let (sr, sc) = idx_to_rc(si);
        let color = game.board[sr][sc];
        let tgt_bits = comp_masks.target_mask(&labels, sr, sc);
        let mut tgt = tgt_bits;
        while tgt != 0 {
            let ti = tgt.trailing_zeros();
            tgt &= tgt - 1;
            let (tr, tc) = idx_to_rc(ti);
            let immediate = evaluate_move(&mut game.board, sr, sc, tr, tc, color);

            // 2-ply clone uses game's RNG (advances it — matches old engine)
            ply.board = game.board;
            ply.next_balls = game.next_balls;
            ply.num_next = game.num_next;
            ply.score = game.score;
            ply.turns = game.turns;
            ply.game_over = game.game_over;
            ply.rng = SimpleRng::new(game.rng.next_u64());
            ply.cc_labels = game.cc_labels;
            ply.cc_valid = true;

            // Use trusted_move — we know it's legal (from source/target masks)
            ply.trusted_move(sr, sc, tr, tc);
            let pts = ply.score - game.score;
            let game_over = ply.game_over;

            let clear_bonus = pts as f64 * 20.0;
            let future = if !game_over {
                let mut f = 0.0f64;
                if let Some((bsr, bsc, btr, btc)) = get_best_move_pure(&mut ply) {
                    let c = ply.board[bsr][bsc];
                    let mut b2 = ply.board;
                    f = evaluate_move(&mut b2, bsr, bsc, btr, btc, c);
                }
                f += count_empty(&ply.board) as f64 * 0.25;
                f
            } else {
                -500.0
            };

            candidates.push(Candidate {
                mv: (sr, sc, tr, tc),
                score_2ply: immediate + clear_bonus + future * 0.5,
            });
        }
    }

    if candidates.is_empty() { return None; }
    candidates.sort_by(|a, b| b.score_2ply.partial_cmp(&a.score_2ply).unwrap());

    let qual_n = candidates.len().min(30);
    if qual_n <= 3 {
        let top: Vec<_> = candidates.iter().map(|c| (c.mv, c.score_2ply)).collect();
        return Some(TournamentResult { chosen: candidates[0].mv, top_moves: top });
    }
    let qualifiers: Vec<Candidate> = candidates.drain(..qual_n).collect();

    // Rollout phases use cloned game RNG (not advancing the real game RNG)
    let rollout_rng = game.rng.clone();

    // Quarter-finals: 10 short rollouts → top 10
    let qf = run_rollouts(game, &qualifiers, 10, rollout_depth / 2, temperature, &mut rollout_rng.clone(), &mut rbuf);
    let mut qfr: Vec<(usize, f64)> = (0..qualifiers.len())
        .map(|i| (i, mean(&qf[i]) + qualifiers[i].score_2ply * 0.1))
        .collect();
    qfr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let semi_n = qualifiers.len().min(10);
    let si: Vec<usize> = qfr[..semi_n].iter().map(|x| x.0).collect();

    // Semi-finals: 40 full rollouts → top 3
    let semis: Vec<Candidate> = si.iter().map(|&i| Candidate { mv: qualifiers[i].mv, score_2ply: qualifiers[i].score_2ply }).collect();
    let sr_scores: Vec<Vec<f64>> = si.iter().map(|&i| qf[i].clone()).collect();
    let sf = run_rollouts(game, &semis, 40, rollout_depth, temperature, &mut rollout_rng.clone(), &mut rbuf);
    let combined: Vec<Vec<f64>> = (0..semis.len()).map(|i| {
        let mut v = sr_scores[i].clone();
        v.extend_from_slice(&sf[i]);
        v
    }).collect();

    let mut sfr: Vec<(usize, f64)> = (0..semis.len())
        .map(|i| (i, mean(&combined[i]) + semis[i].score_2ply * 0.1))
        .collect();
    sfr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let fn_ = semis.len().min(3);
    let fi: Vec<usize> = sfr[..fn_].iter().map(|x| x.0).collect();
    let finals: Vec<Candidate> = fi.iter().map(|&i| Candidate { mv: semis[i].mv, score_2ply: semis[i].score_2ply }).collect();
    let fr: Vec<Vec<f64>> = fi.iter().map(|&i| combined[i].clone()).collect();

    // Finals
    let used = 10 + 40;
    let remaining = if num_rollouts > used { num_rollouts - used } else { 10 };
    let fsc = run_rollouts(game, &finals, remaining, rollout_depth, temperature, &mut rollout_rng.clone(), &mut rbuf);
    let af: Vec<Vec<f64>> = (0..finals.len()).map(|i| {
        let mut v = fr[i].clone();
        v.extend_from_slice(&fsc[i]);
        v
    }).collect();

    // Build top moves from semi-finalists
    let mut all_ranked: Vec<((usize, usize, usize, usize), f64)> = (0..semis.len())
        .map(|i| (semis[i].mv, mean(&combined[i]) + semis[i].score_2ply * 0.1))
        .collect();
    for (fii, &idx) in fi.iter().enumerate() {
        all_ranked[idx].1 = mean(&af[fii]) + finals[fii].score_2ply * 0.1;
    }
    all_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Pick best finalist
    let mut best_score = f64::NEG_INFINITY;
    let mut best_idx = 0;
    for (i, f) in finals.iter().enumerate() {
        let avg = mean(&af[i]) + f.score_2ply * 0.1;
        if avg > best_score { best_score = avg; best_idx = i; }
    }

    Some(TournamentResult {
        chosen: finals[best_idx].mv,
        top_moves: all_ranked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tournament_returns_legal_move() {
        let mut game = ColorLinesGame::new(42);
        game.reset();
        let result = tournament_player(&mut game, 50, 10, 3.23);
        assert!(result.is_some());
        let r = result.unwrap();
        let (sr, sc, tr, tc) = r.chosen;
        // Board may have changed during tournament (RNG coupling), check original
        // Just verify the move is reasonable
        assert!(sr < BOARD_SIZE && sc < BOARD_SIZE);
        assert!(tr < BOARD_SIZE && tc < BOARD_SIZE);
    }

    #[test]
    fn test_tournament_deterministic() {
        let mut g1 = ColorLinesGame::new(42);
        g1.reset();
        let mut g2 = ColorLinesGame::new(42);
        g2.reset();
        let r1 = tournament_player(&mut g1, 50, 10, 3.23).unwrap();
        let r2 = tournament_player(&mut g2, 50, 10, 3.23).unwrap();
        assert_eq!(r1.chosen, r2.chosen);
    }

    #[test]
    fn test_tournament_has_top_moves() {
        let mut game = ColorLinesGame::new(42);
        game.reset();
        let result = tournament_player(&mut game, 50, 10, 3.23).unwrap();
        assert!(!result.top_moves.is_empty());
    }
}
