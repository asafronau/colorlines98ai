// Color Lines 98 game state and logic.

use crate::board::*;
use crate::rng::SimpleRng;

#[derive(Clone)]
pub struct ColorLinesGame {
    pub board: Board,
    pub next_balls: Vec<((usize, usize), i8)>, // ((row, col), color)
    pub score: i32,
    pub turns: i32,
    pub game_over: bool,
    pub rng: SimpleRng,
    num_colors: i8,
}

impl ColorLinesGame {
    pub fn new(seed: u64) -> Self {
        ColorLinesGame {
            board: [[0i8; BOARD_SIZE]; BOARD_SIZE],
            next_balls: Vec::new(),
            score: 0,
            turns: 0,
            game_over: false,
            rng: SimpleRng::new(seed),
            num_colors: NUM_COLORS,
        }
    }

    pub fn reset(&mut self) {
        self.board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        self.score = 0;
        self.turns = 0;
        self.game_over = false;
        self.generate_next_balls();
        self.spawn_balls();
        self.generate_next_balls();
    }

    /// Deep clone with a replacement RNG.
    pub fn clone_with_rng(&self, rng: SimpleRng) -> Self {
        ColorLinesGame {
            board: self.board,
            next_balls: self.next_balls.clone(),
            score: self.score,
            turns: self.turns,
            game_over: self.game_over,
            rng,
            num_colors: self.num_colors,
        }
    }

    // ── Ball spawning ─────────────────────────────────────────────

    fn generate_next_balls(&mut self) {
        let empty = get_empty_cells(&self.board);
        let n_empty = empty.len();
        if n_empty == 0 {
            self.next_balls = Vec::new();
            return;
        }
        let n = BALLS_PER_TURN.min(n_empty);
        let indices = self.rng.choice_no_replace(n_empty, n);
        let colors = self.rng.integers(1, self.num_colors as i64 + 1, n);
        self.next_balls = (0..n)
            .map(|i| (empty[indices[i]], colors[i] as i8))
            .collect();
    }

    fn spawn_balls(&mut self) {
        for i in 0..self.next_balls.len() {
            let ((row, col), color) = self.next_balls[i];
            if self.board[row][col] == 0 {
                self.board[row][col] = color;
            } else {
                let empty = get_empty_cells(&self.board);
                if !empty.is_empty() {
                    let idx = self.rng.randint(0, empty.len() as i64) as usize;
                    self.board[empty[idx].0][empty[idx].1] = color;
                }
            }
        }
    }

    // ── Move execution ────────────────────────────────────────────

    /// Execute a move with full validation.
    /// Returns (valid, score_gained, cleared, game_over).
    pub fn move_ball(
        &mut self,
        sr: usize,
        sc: usize,
        tr: usize,
        tc: usize,
    ) -> (bool, i32, usize, bool) {
        if self.game_over {
            return (false, 0, 0, true);
        }
        if self.board[sr][sc] == 0 || self.board[tr][tc] != 0 {
            return (false, 0, 0, self.game_over);
        }

        let labels = label_empty_components(&self.board);
        if !is_reachable(&labels, sr, sc, tr, tc) {
            return (false, 0, 0, self.game_over);
        }

        // Execute move
        let color = self.board[sr][sc];
        self.board[sr][sc] = 0;
        self.board[tr][tc] = color;
        self.turns += 1;

        let mut total_score = 0i32;
        let mut total_cleared = 0usize;

        let cleared = clear_lines_at(&mut self.board, tr, tc);
        if cleared > 0 {
            let pts = calculate_score(cleared);
            self.score += pts;
            total_score += pts;
            total_cleared += cleared;
        } else {
            // No line → spawn + check spawned balls for lines
            self.spawn_balls();
            let next_balls_snapshot: Vec<_> = self.next_balls.clone();
            for &((br, bc), _) in &next_balls_snapshot {
                if self.board[br][bc] != 0 {
                    let spawn_cleared = clear_lines_at(&mut self.board, br, bc);
                    if spawn_cleared > 0 {
                        let pts = calculate_score(spawn_cleared);
                        self.score += pts;
                        total_score += pts;
                        total_cleared += spawn_cleared;
                    }
                }
            }
            self.generate_next_balls();
            if count_empty(&self.board) == 0 {
                self.game_over = true;
            }
        }

        (true, total_score, total_cleared, self.game_over)
    }

    /// Execute a move known to be legal (skip validation). For MCTS hot path.
    pub fn trusted_move(&mut self, sr: usize, sc: usize, tr: usize, tc: usize) {
        let color = self.board[sr][sc];
        self.board[sr][sc] = 0;
        self.board[tr][tc] = color;
        self.turns += 1;

        let cleared = clear_lines_at(&mut self.board, tr, tc);
        if cleared > 0 {
            self.score += calculate_score(cleared);
        } else {
            self.spawn_balls();
            let next_balls_snapshot: Vec<_> = self.next_balls.clone();
            for &((br, bc), _) in &next_balls_snapshot {
                if self.board[br][bc] != 0 {
                    let spawn_cleared = clear_lines_at(&mut self.board, br, bc);
                    if spawn_cleared > 0 {
                        self.score += calculate_score(spawn_cleared);
                    }
                }
            }
            self.generate_next_balls();
            if count_empty(&self.board) == 0 {
                self.game_over = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let g = ColorLinesGame::new(42);
        assert_eq!(g.score, 0);
        assert_eq!(g.turns, 0);
        assert!(!g.game_over);
    }

    #[test]
    fn test_reset_places_balls() {
        let mut g = ColorLinesGame::new(42);
        g.reset();
        let n_balls = 81 - count_empty(&g.board);
        assert_eq!(n_balls, 3); // 3 balls spawned from first generate+spawn
        assert_eq!(g.next_balls.len(), 3); // 3 next balls generated
    }

    #[test]
    fn test_deterministic_reset() {
        let mut g1 = ColorLinesGame::new(42);
        g1.reset();
        let mut g2 = ColorLinesGame::new(42);
        g2.reset();
        assert_eq!(g1.board, g2.board);
        assert_eq!(g1.next_balls, g2.next_balls);
    }

    #[test]
    fn test_clone_independent() {
        let mut g = ColorLinesGame::new(42);
        g.reset();
        let mut g2 = g.clone();
        // Mutate clone
        g2.board[0][0] = 5;
        assert_ne!(g.board[0][0], g2.board[0][0]);
    }
}
