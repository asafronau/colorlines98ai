/// Tournament bracket player — successive halving with heuristic rollouts.
///
/// Phase 1: 2-ply evaluation for all legal moves
/// Phase 2: Quarter-finals (top 30, 10 short rollouts → top 10)
/// Phase 3: Semi-finals (40 full rollouts → top 3)
/// Phase 4: Finals (remaining rollouts → best move)

use crate::board::*;
use crate::game::ColorLinesGame;
use crate::heuristic::{evaluate_move, get_best_move, get_softmax_move};
use crate::rng::SimpleRng;

/// Result of tournament play: chosen move + top candidates with scores.
pub struct TournamentResult {
    pub chosen: (usize, usize, usize, usize),
    pub top_moves: Vec<((usize, usize, usize, usize), f64)>,
}

struct Candidate {
    mv: (usize, usize, usize, usize),
    score_2ply: f64,
}

fn rollout(
    game: &ColorLinesGame,
    mv: (usize, usize, usize, usize),
    depth: usize,
    temperature: f64,
    seed: u64,
) -> f64 {
    let mut clone = game.clone_with_rng(SimpleRng::new(seed));
    let (valid, _, _, _) = clone.move_ball(mv.0, mv.1, mv.2, mv.3);
    if !valid {
        return -1000.0;
    }
    let base_score = clone.score;

    let mut rng = SimpleRng::new(seed.wrapping_add(1));
    for _ in 0..depth {
        if clone.game_over {
            break;
        }
        let m = if temperature > 0.0 {
            get_softmax_move(&clone, temperature, &mut rng)
        } else {
            get_best_move(&clone)
        };
        match m {
            Some((sr, sc, tr, tc)) => {
                clone.move_ball(sr, sc, tr, tc);
            }
            None => break,
        }
    }
    (clone.score - base_score) as f64
}

fn run_rollouts(
    game: &ColorLinesGame,
    candidates: &[Candidate],
    n_rollouts: usize,
    depth: usize,
    temperature: f64,
    rng: &mut SimpleRng,
) -> Vec<Vec<f64>> {
    candidates
        .iter()
        .map(|c| {
            (0..n_rollouts)
                .map(|_| rollout(game, c.mv, depth, temperature, rng.next_u64()))
                .collect()
        })
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Play one tournament move with successive halving.
pub fn tournament_player(
    game: &ColorLinesGame,
    num_rollouts: usize,
    rollout_depth: usize,
    temperature: f64,
    rng: &mut SimpleRng,
) -> Option<TournamentResult> {
    let source_mask = get_source_mask(&game.board);
    let labels = label_empty_components(&game.board);

    // Phase 1: 2-ply for ALL legal moves
    let mut candidates: Vec<Candidate> = Vec::with_capacity(256);
    let mut board_copy = game.board;

    for sr in 0..BOARD_SIZE {
        for sc in 0..BOARD_SIZE {
            if source_mask[sr][sc] == 0 {
                continue;
            }
            let color = game.board[sr][sc];
            let target_mask = get_target_mask(&labels, sr, sc);
            for tr in 0..BOARD_SIZE {
                for tc in 0..BOARD_SIZE {
                    if target_mask[tr][tc] == 0 {
                        continue;
                    }
                    let immediate = evaluate_move(&mut board_copy, sr, sc, tr, tc, color);

                    // 2-ply: simulate move, evaluate best response
                    let mut ply_game = game.clone_with_rng(SimpleRng::new(rng.next_u64()));
                    let (valid, _, _, _) = ply_game.move_ball(sr, sc, tr, tc);
                    if !valid {
                        continue;
                    }

                    let pts = ply_game.score - game.score;
                    let clear_bonus = pts as f64 * 20.0;
                    let future = if !ply_game.game_over {
                        let mut f = 0.0f64;
                        if let Some((bsr, bsc, btr, btc)) = get_best_move(&ply_game) {
                            let c = ply_game.board[bsr][bsc];
                            let mut b2 = ply_game.board;
                            f = evaluate_move(&mut b2, bsr, bsc, btr, btc, c);
                        }
                        f += count_empty(&ply_game.board) as f64 * 0.25;
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
        }
    }

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| b.score_2ply.partial_cmp(&a.score_2ply).unwrap());

    // Early exit for few candidates
    let qual_n = candidates.len().min(30);
    if qual_n <= 3 {
        let top: Vec<_> = candidates
            .iter()
            .map(|c| (c.mv, c.score_2ply))
            .collect();
        return Some(TournamentResult {
            chosen: candidates[0].mv,
            top_moves: top,
        });
    }
    let qualifiers: Vec<Candidate> = candidates.drain(..qual_n).collect();

    // Phase 2: Quarter-finals — 10 short rollouts → top 10
    let qf_scores = run_rollouts(game, &qualifiers, 10, rollout_depth / 2, temperature, rng);
    let mut qf_ranked: Vec<(usize, f64)> = (0..qualifiers.len())
        .map(|i| (i, mean(&qf_scores[i]) + qualifiers[i].score_2ply * 0.1))
        .collect();
    qf_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let semi_n = qualifiers.len().min(10);
    let semi_indices: Vec<usize> = qf_ranked[..semi_n].iter().map(|x| x.0).collect();

    // Phase 3: Semi-finals — 40 full rollouts → top 3
    let semis: Vec<Candidate> = semi_indices
        .iter()
        .map(|&i| Candidate {
            mv: qualifiers[i].mv,
            score_2ply: qualifiers[i].score_2ply,
        })
        .collect();
    let semi_raw: Vec<Vec<f64>> = semi_indices.iter().map(|&i| qf_scores[i].clone()).collect();
    let sf_scores = run_rollouts(game, &semis, 40, rollout_depth, temperature, rng);

    let combined: Vec<Vec<f64>> = (0..semis.len())
        .map(|i| {
            let mut v = semi_raw[i].clone();
            v.extend_from_slice(&sf_scores[i]);
            v
        })
        .collect();

    let mut sf_ranked: Vec<(usize, f64)> = (0..semis.len())
        .map(|i| (i, mean(&combined[i]) + semis[i].score_2ply * 0.1))
        .collect();
    sf_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let final_n = semis.len().min(3);
    let final_indices: Vec<usize> = sf_ranked[..final_n].iter().map(|x| x.0).collect();
    let finals: Vec<Candidate> = final_indices
        .iter()
        .map(|&i| Candidate {
            mv: semis[i].mv,
            score_2ply: semis[i].score_2ply,
        })
        .collect();
    let finals_raw: Vec<Vec<f64>> = final_indices.iter().map(|&i| combined[i].clone()).collect();

    // Phase 4: Finals — remaining rollouts
    let used = 10 + 40;
    let remaining = if num_rollouts > used {
        num_rollouts - used
    } else {
        10
    };
    let final_scores = run_rollouts(game, &finals, remaining, rollout_depth, temperature, rng);

    let all_finals: Vec<Vec<f64>> = (0..finals.len())
        .map(|i| {
            let mut v = finals_raw[i].clone();
            v.extend_from_slice(&final_scores[i]);
            v
        })
        .collect();

    // Build top moves list from semi-finalists
    let mut all_ranked: Vec<((usize, usize, usize, usize), f64)> = (0..semis.len())
        .map(|i| (semis[i].mv, mean(&combined[i]) + semis[i].score_2ply * 0.1))
        .collect();
    for (fi, &idx) in final_indices.iter().enumerate() {
        all_ranked[idx].1 = mean(&all_finals[fi]) + finals[fi].score_2ply * 0.1;
    }
    all_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Pick best finalist
    let mut best_score = f64::NEG_INFINITY;
    let mut best_idx = 0;
    for (i, f) in finals.iter().enumerate() {
        let avg = mean(&all_finals[i]) + f.score_2ply * 0.1;
        if avg > best_score {
            best_score = avg;
            best_idx = i;
        }
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
        let mut rng = SimpleRng::new(42);
        let result = tournament_player(&game, 50, 10, 3.23, &mut rng);
        assert!(result.is_some());
        let r = result.unwrap();
        let (sr, sc, tr, tc) = r.chosen;
        assert!(game.board[sr][sc] != 0);
        assert!(game.board[tr][tc] == 0);
    }

    #[test]
    fn test_tournament_deterministic() {
        let mut game = ColorLinesGame::new(42);
        game.reset();
        let mut rng1 = SimpleRng::new(99);
        let mut rng2 = SimpleRng::new(99);
        let r1 = tournament_player(&game, 50, 10, 3.23, &mut rng1).unwrap();
        let r2 = tournament_player(&game, 50, 10, 3.23, &mut rng2).unwrap();
        assert_eq!(r1.chosen, r2.chosen);
    }

    #[test]
    fn test_tournament_has_top_moves() {
        let mut game = ColorLinesGame::new(42);
        game.reset();
        let mut rng = SimpleRng::new(42);
        let result = tournament_player(&game, 50, 10, 3.23, &mut rng).unwrap();
        assert!(!result.top_moves.is_empty());
        // Chosen move should be in top moves
        assert!(result.top_moves.iter().any(|(m, _)| *m == result.chosen));
    }
}
