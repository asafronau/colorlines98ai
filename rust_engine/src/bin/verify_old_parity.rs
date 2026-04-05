/// Verify parity with old Rust engine by using its exact xorshift64 RNG.
///
/// Plays games with the old RNG + old game logic flow and compares scores
/// against the old engine's datagen output.

use colorlines98::board::*;
use colorlines98::heuristic::evaluate_move;
use colorlines98::rng::Xorshift64;

const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

/// Minimal game struct matching old engine exactly.
struct OldGame {
    board: Board,
    next_balls: [(usize, usize, i8); BALLS_PER_TURN], // (row, col, color)
    num_next: usize,
    score: i32,
    turns: i32,
    game_over: bool,
    rng: Xorshift64,
}

impl Clone for OldGame {
    fn clone(&self) -> Self {
        OldGame {
            board: self.board,
            next_balls: self.next_balls,
            num_next: self.num_next,
            score: self.score,
            turns: self.turns,
            game_over: self.game_over,
            rng: self.rng.clone(),
        }
    }
}

impl OldGame {
    fn new(seed: u64) -> Self {
        OldGame {
            board: [[0i8; BOARD_SIZE]; BOARD_SIZE],
            next_balls: [(0, 0, 0); BALLS_PER_TURN],
            num_next: 0,
            score: 0,
            turns: 0,
            game_over: false,
            rng: Xorshift64::new(seed),
        }
    }

    fn reset(&mut self) {
        self.board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        self.score = 0;
        self.turns = 0;
        self.game_over = false;
        self.generate_next_balls();
        self.spawn_balls();
        self.generate_next_balls();
    }

    fn count_empty(&self) -> usize {
        count_empty(&self.board)
    }

    fn get_empty_cells(&self) -> Vec<(usize, usize)> {
        get_empty_cells(&self.board)
    }

    fn generate_next_balls(&mut self) {
        let empty = self.get_empty_cells();
        if empty.is_empty() {
            self.num_next = 0;
            return;
        }
        let n = BALLS_PER_TURN.min(empty.len());
        // Fisher-Yates partial shuffle — matching old code exactly
        let mut buf = empty;
        for i in 0..n {
            let j = i + self.rng.next_usize(buf.len() - i);
            buf.swap(i, j);
        }
        for i in 0..n {
            self.next_balls[i] = (
                buf[i].0,
                buf[i].1,
                1 + self.rng.next_usize(NUM_COLORS as usize) as i8,
            );
        }
        self.num_next = n;
    }

    fn spawn_balls(&mut self) {
        for i in 0..self.num_next {
            let (row, col, color) = self.next_balls[i];
            if self.board[row][col] == 0 {
                self.board[row][col] = color;
            } else {
                let empty = self.get_empty_cells();
                if !empty.is_empty() {
                    let idx = self.rng.next_usize(empty.len());
                    self.board[empty[idx].0][empty[idx].1] = color;
                }
            }
        }
    }

    fn make_move(&mut self, sr: usize, sc: usize, tr: usize, tc: usize) -> (bool, i32) {
        if self.game_over { return (false, 0); }
        if self.board[sr][sc] == 0 || self.board[tr][tc] != 0 { return (false, 0); }

        let labels = label_empty_components(&self.board);
        if !is_reachable(&labels, sr, sc, tr, tc) { return (false, 0); }

        let color = self.board[sr][sc];
        self.board[sr][sc] = 0;
        self.board[tr][tc] = color;
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
            if self.count_empty() == 0 {
                self.game_over = true;
            }
        }
        (true, total_pts)
    }

    fn get_best_move(&self) -> Option<(usize, usize, usize, usize)> {
        let source_mask = get_source_mask(&self.board);
        let labels = label_empty_components(&self.board);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_move = None;
        let mut board = self.board;

        for sr in 0..BOARD_SIZE {
            for sc in 0..BOARD_SIZE {
                if source_mask[sr][sc] == 0 { continue; }
                let color = board[sr][sc];
                let tmask = get_target_mask(&labels, sr, sc);
                for tr in 0..BOARD_SIZE {
                    for tc in 0..BOARD_SIZE {
                        if tmask[tr][tc] == 0 { continue; }
                        let s = evaluate_move(&mut board, sr, sc, tr, tc, color);
                        if s > best_score {
                            best_score = s;
                            best_move = Some((sr, sc, tr, tc));
                        }
                    }
                }
            }
        }
        best_move
    }
}

fn play_heuristic_game(seed: u64) -> (i32, i32) {
    let mut g = OldGame::new(seed);
    g.reset();
    while !g.game_over {
        match g.get_best_move() {
            Some((sr, sc, tr, tc)) => { g.make_move(sr, sc, tr, tc); }
            None => break,
        }
    }
    (g.score, g.turns)
}

fn main() {
    println!("=== Parity check: new engine with old xorshift64 RNG ===\n");

    // Play 20 heuristic games and print scores
    let mut total = 0i64;
    for seed in 0..20u64 {
        let (score, turns) = play_heuristic_game(seed);
        total += score as i64;
        println!("  seed={:2}: score={:5} turns={:3}", seed, score, turns);
    }
    println!("\n  Mean: {:.0}", total as f64 / 20.0);
    println!("\nCompare these with: cd _old_rust && cargo run --release -- bench");
    println!("If heuristic scores match → game engine is identical.");
    println!("Any differences → bug in board logic or RNG consumption order.");
}
