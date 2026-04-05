/// Verify tournament parity with old engine.
///
/// Uses xorshift64 RNG + exact old tournament logic (RNG coupling).
/// Target: seed=12 should produce score=1018, turns=489.

use colorlines98::board::*;
use colorlines98::heuristic::evaluate_move;
use colorlines98::rng::Xorshift64;

// ── Old-style game with mutable RNG coupling ────────────────────────

#[derive(Clone)]
struct OldGame {
    board: Board,
    next_balls: [(usize, usize, i8); BALLS_PER_TURN],
    num_next: usize,
    score: i32,
    turns: i32,
    game_over: bool,
    rng: Xorshift64,
    cc_labels: Board,
    cc_valid: bool,
}

impl OldGame {
    fn new(seed: u64) -> Self {
        OldGame {
            board: [[0; BOARD_SIZE]; BOARD_SIZE],
            next_balls: [(0, 0, 0); BALLS_PER_TURN],
            num_next: 0, score: 0, turns: 0, game_over: false,
            rng: Xorshift64::new(seed),
            cc_labels: [[0; BOARD_SIZE]; BOARD_SIZE],
            cc_valid: false,
        }
    }

    fn reset(&mut self) {
        self.board = [[0; BOARD_SIZE]; BOARD_SIZE];
        self.score = 0; self.turns = 0; self.game_over = false; self.cc_valid = false;
        self.generate_next_balls();
        self.spawn_balls();
        self.generate_next_balls();
    }

    fn clone_with_seed(&self, seed: u64) -> Self {
        let mut c = self.clone();
        c.rng = Xorshift64::new(seed);
        c.cc_valid = false;
        c
    }

    fn ensure_cc(&mut self) {
        if self.cc_valid { return; }
        self.cc_labels = label_empty_components(&self.board);
        self.cc_valid = true;
    }

    fn generate_next_balls(&mut self) {
        let empty = get_empty_cells(&self.board);
        if empty.is_empty() { self.num_next = 0; return; }
        let n = BALLS_PER_TURN.min(empty.len());
        let mut buf = empty;
        for i in 0..n {
            let j = i + self.rng.next_usize(buf.len() - i);
            buf.swap(i, j);
        }
        for i in 0..n {
            self.next_balls[i] = (buf[i].0, buf[i].1, 1 + self.rng.next_usize(NUM_COLORS as usize) as i8);
        }
        self.num_next = n;
    }

    fn spawn_balls(&mut self) {
        for i in 0..self.num_next {
            let (row, col, color) = self.next_balls[i];
            if self.board[row][col] == 0 {
                self.board[row][col] = color;
            } else {
                let empty = get_empty_cells(&self.board);
                if !empty.is_empty() {
                    let idx = self.rng.next_usize(empty.len());
                    self.board[empty[idx].0][empty[idx].1] = color;
                }
            }
        }
        self.cc_valid = false;
    }

    fn make_move(&mut self, sr: usize, sc: usize, tr: usize, tc: usize) -> (bool, i32) {
        if self.game_over { return (false, 0); }
        if self.board[sr][sc] == 0 || self.board[tr][tc] != 0 { return (false, 0); }
        self.ensure_cc();
        if !is_reachable(&self.cc_labels, sr, sc, tr, tc) { return (false, 0); }

        let color = self.board[sr][sc];
        self.board[sr][sc] = 0;
        self.board[tr][tc] = color;
        self.cc_valid = false;
        self.turns += 1;

        let mut total_pts = 0i32;
        let cleared = clear_lines_at(&mut self.board, tr, tc);
        if cleared > 0 {
            let pts = calculate_score(cleared);
            self.score += pts;
            total_pts = pts;
        } else {
            let saved = self.next_balls;
            let saved_n = self.num_next;
            self.spawn_balls();
            for i in 0..saved_n {
                let (br, bc, _) = saved[i];
                if self.board[br][bc] != 0 {
                    let sc = clear_lines_at(&mut self.board, br, bc);
                    if sc > 0 {
                        let pts = calculate_score(sc as usize);
                        self.score += pts;
                        total_pts += pts;
                    }
                }
            }
            self.generate_next_balls();
            if count_empty(&self.board) == 0 { self.game_over = true; }
        }
        (true, total_pts)
    }

    fn get_source_mask(&self) -> [bool; 81] {
        let mut mask = [false; 81];
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                if self.board[r][c] == 0 { continue; }
                for &(dr, dc) in &[(0i32,1i32),(0,-1),(1,0),(-1,0)] {
                    let (nr, nc) = (r as i32 + dr, c as i32 + dc);
                    if nr >= 0 && nr < 9 && nc >= 0 && nc < 9 && self.board[nr as usize][nc as usize] == 0 {
                        mask[r * 9 + c] = true; break;
                    }
                }
            }
        }
        mask
    }

    fn get_target_mask(&self, sr: usize, sc: usize) -> [bool; 81] {
        let mut mask = [false; 81];
        let mut labels = [0i8; 4];
        let mut n_labels = 0;
        for &(dr, dc) in &[(0i32,1i32),(0,-1),(1,0),(-1,0)] {
            let (nr, nc) = (sr as i32 + dr, sc as i32 + dc);
            if nr >= 0 && nr < 9 && nc >= 0 && nc < 9 {
                let lbl = self.cc_labels[nr as usize][nc as usize];
                if lbl > 0 {
                    let mut found = false;
                    for i in 0..n_labels { if labels[i] == lbl { found = true; break; } }
                    if !found { labels[n_labels] = lbl; n_labels += 1; }
                }
            }
        }
        for r in 0..9 { for c in 0..9 {
            if self.cc_labels[r][c] > 0 {
                for i in 0..n_labels {
                    if self.cc_labels[r][c] == labels[i] { mask[r*9+c] = true; break; }
                }
            }
        }}
        mask
    }
}

// ── Old-style heuristic (uses game.rng for softmax) ─────────────────

fn get_best_move(game: &mut OldGame) -> Option<(usize, usize, usize, usize)> {
    game.ensure_cc();
    let sm = game.get_source_mask();
    let mut best_score = f64::NEG_INFINITY;
    let mut best = None;
    for sr in 0..9 { for sc in 0..9 {
        if !sm[sr*9+sc] { continue; }
        let color = game.board[sr][sc];
        let tm = game.get_target_mask(sr, sc);
        for tr in 0..9 { for tc in 0..9 {
            if !tm[tr*9+tc] { continue; }
            let s = evaluate_move(&mut game.board, sr, sc, tr, tc, color);
            if s > best_score { best_score = s; best = Some((sr, sc, tr, tc)); }
        }}
    }}
    best
}

fn get_softmax_move(game: &mut OldGame, temperature: f64) -> Option<(usize, usize, usize, usize)> {
    game.ensure_cc();
    let sm = game.get_source_mask();
    let mut moves = Vec::with_capacity(256);
    let mut scores = Vec::with_capacity(256);
    for sr in 0..9 { for sc in 0..9 {
        if !sm[sr*9+sc] { continue; }
        let color = game.board[sr][sc];
        let tm = game.get_target_mask(sr, sc);
        for tr in 0..9 { for tc in 0..9 {
            if !tm[tr*9+tc] { continue; }
            let s = evaluate_move(&mut game.board, sr, sc, tr, tc, color);
            moves.push((sr, sc, tr, tc));
            scores.push(s);
        }}
    }}
    if moves.is_empty() { return None; }
    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = scores.iter().map(|s| ((s - max_s) / temperature).exp()).collect();
    let sum: f64 = probs.iter().sum();
    for p in probs.iter_mut() { *p /= sum; }
    // Key: uses game.rng, not separate rng
    let r = game.rng.next_f64();
    let mut cumul = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumul += p;
        if r <= cumul { return Some(moves[i]); }
    }
    Some(*moves.last().unwrap())
}

// ── Old-style tournament ────────────────────────────────────────────

fn rollout(game: &OldGame, sr: usize, sc: usize, tr: usize, tc: usize,
           depth: usize, temperature: f64, seed: u64) -> f64 {
    let mut clone = game.clone_with_seed(seed);
    let (valid, pts) = clone.make_move(sr, sc, tr, tc);
    if !valid { return -1000.0; }
    let base_score = clone.score;
    for _ in 0..depth {
        if clone.game_over { break; }
        let m = if temperature > 0.0 {
            get_softmax_move(&mut clone, temperature)
        } else {
            get_best_move(&mut clone)
        };
        match m {
            Some((a, b, c, d)) => { clone.make_move(a, b, c, d); }
            None => break,
        }
    }
    (clone.score - base_score) as f64 + pts as f64
}

struct Candidate {
    mv: (usize, usize, usize, usize),
    score_2ply: f64,
}

fn run_rollouts(game: &OldGame, candidates: &[Candidate], n: usize, depth: usize,
                temp: f64, rng: &mut Xorshift64) -> Vec<Vec<f64>> {
    candidates.iter().map(|c| {
        (0..n).map(|_| rollout(game, c.mv.0, c.mv.1, c.mv.2, c.mv.3, depth, temp, rng.next_u64())).collect()
    }).collect()
}

fn mean(v: &[f64]) -> f64 { if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 } }

fn tournament_move(game: &mut OldGame, num_rollouts: usize, depth: usize, temp: f64)
    -> Option<(usize, usize, usize, usize)>
{
    game.ensure_cc();
    let sm = game.get_source_mask();
    let mut candidates: Vec<Candidate> = Vec::with_capacity(256);
    let mut ply = game.clone();

    for sr in 0..9 { for sc in 0..9 {
        if !sm[sr*9+sc] { continue; }
        let color = game.board[sr][sc];
        let tm = game.get_target_mask(sr, sc);
        for tr in 0..9 { for tc in 0..9 {
            if !tm[tr*9+tc] { continue; }
            let immediate = evaluate_move(&mut game.board, sr, sc, tr, tc, color);
            // Clone into ply (reuse allocation pattern from old code)
            ply.board = game.board;
            ply.next_balls = game.next_balls;
            ply.num_next = game.num_next;
            ply.score = game.score;
            ply.turns = game.turns;
            ply.game_over = game.game_over;
            ply.rng = Xorshift64::new(game.rng.next_u64());
            ply.cc_valid = false;
            let (valid, pts) = ply.make_move(sr, sc, tr, tc);
            if !valid { continue; }
            let clear_bonus = pts as f64 * 20.0;
            let future = if !ply.game_over {
                let mut f = 0.0f64;
                if let Some((a,b,c,d)) = get_best_move(&mut ply) {
                    let col = ply.board[a][b];
                    f = evaluate_move(&mut ply.board, a, b, c, d, col);
                }
                f += count_empty(&ply.board) as f64 * 0.25;
                f
            } else { -500.0 };
            candidates.push(Candidate { mv: (sr,sc,tr,tc), score_2ply: immediate + clear_bonus + future * 0.5 });
        }}
    }}

    if candidates.is_empty() { return None; }
    candidates.sort_by(|a, b| b.score_2ply.partial_cmp(&a.score_2ply).unwrap());

    let qual_n = candidates.len().min(30);
    let qualifiers: Vec<Candidate> = candidates.drain(..qual_n).collect();
    if qualifiers.len() <= 3 { return Some(qualifiers[0].mv); }

    let qf = run_rollouts(game, &qualifiers, 10, depth / 2, temp, &mut game.rng.clone());
    let mut qfr: Vec<(usize, f64)> = (0..qualifiers.len()).map(|i| (i, mean(&qf[i]) + qualifiers[i].score_2ply * 0.1)).collect();
    qfr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let semi_n = qualifiers.len().min(10);
    let si: Vec<usize> = qfr[..semi_n].iter().map(|x| x.0).collect();

    let semis: Vec<Candidate> = si.iter().map(|&i| Candidate { mv: qualifiers[i].mv, score_2ply: qualifiers[i].score_2ply }).collect();
    let sr: Vec<Vec<f64>> = si.iter().map(|&i| qf[i].clone()).collect();
    let sf = run_rollouts(game, &semis, 40, depth, temp, &mut game.rng.clone());
    let combined: Vec<Vec<f64>> = (0..semis.len()).map(|i| { let mut v = sr[i].clone(); v.extend_from_slice(&sf[i]); v }).collect();

    let mut sfr: Vec<(usize, f64)> = (0..semis.len()).map(|i| (i, mean(&combined[i]) + semis[i].score_2ply * 0.1)).collect();
    sfr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let fn_ = semis.len().min(3);
    let fi: Vec<usize> = sfr[..fn_].iter().map(|x| x.0).collect();
    let finals: Vec<Candidate> = fi.iter().map(|&i| Candidate { mv: semis[i].mv, score_2ply: semis[i].score_2ply }).collect();
    let fr: Vec<Vec<f64>> = fi.iter().map(|&i| combined[i].clone()).collect();

    let used = 10 + 40;
    let remaining = if num_rollouts > used { num_rollouts - used } else { 10 };
    let fsc = run_rollouts(game, &finals, remaining, depth, temp, &mut game.rng.clone());
    let af: Vec<Vec<f64>> = (0..finals.len()).map(|i| { let mut v = fr[i].clone(); v.extend_from_slice(&fsc[i]); v }).collect();

    let mut best_score = f64::NEG_INFINITY;
    let mut best_idx = 0;
    for (i, f) in finals.iter().enumerate() {
        let avg = mean(&af[i]) + f.score_2ply * 0.1;
        if avg > best_score { best_score = avg; best_idx = i; }
    }
    Some(finals[best_idx].mv)
}

fn main() {
    println!("=== Tournament parity check (xorshift64) ===\n");

    let targets = [(12, 1018, 489), (10, 785, 381), (0, 2519, 1161)];

    for &(seed, expected_score, expected_turns) in &targets {
        let mut g = OldGame::new(seed);
        g.reset();
        let start = std::time::Instant::now();
        while !g.game_over {
            match tournament_move(&mut g, 50, 20, 3.23) {
                Some((sr, sc, tr, tc)) => { g.make_move(sr, sc, tr, tc); }
                None => break,
            }
            if g.turns % 100 == 0 {
                eprint!("  seed={} turn={} score={}\r", seed, g.turns, g.score);
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let ok = g.score == expected_score && g.turns == expected_turns;
        println!("  seed={:2}: score={:5} turns={:4} ({:.1}s) expected=({},{}) {}",
            seed, g.score, g.turns, elapsed, expected_score, expected_turns,
            if ok { "✓ MATCH" } else { "✗ MISMATCH" });
    }
}
