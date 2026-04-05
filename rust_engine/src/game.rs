// Color Lines 98 game state and logic.
//
// Zero-allocation design: all state is inline (no Vec, no heap).
// Clone is a simple memcpy (~100 bytes). Critical for rollout performance.

use crate::board::*;
use crate::rng::SimpleRng;

/// A pending ball spawn: position + color.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct NextBall {
    pub row: u8,
    pub col: u8,
    pub color: i8,
}

/// Full game state — 100% stack-allocated. Clone = memcpy.
#[derive(Clone)]
pub struct ColorLinesGame {
    pub board: Board,
    pub next_balls: [NextBall; BALLS_PER_TURN],
    pub num_next: u8,
    pub score: i32,
    pub turns: i32,
    pub game_over: bool,
    pub rng: SimpleRng,
}

impl ColorLinesGame {
    pub fn new(seed: u64) -> Self {
        ColorLinesGame {
            board: [[0i8; BOARD_SIZE]; BOARD_SIZE],
            next_balls: [NextBall::default(); BALLS_PER_TURN],
            num_next: 0,
            score: 0,
            turns: 0,
            game_over: false,
            rng: SimpleRng::new(seed),
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

    /// Clone with a replacement RNG. Entire struct is Copy-like (no heap).
    #[inline]
    pub fn clone_with_rng(&self, rng: SimpleRng) -> Self {
        let mut c = self.clone();
        c.rng = rng;
        c
    }

    /// Get next_balls as the tuple format used by Python for compatibility.
    pub fn next_balls_tuples(&self) -> Vec<((usize, usize), i8)> {
        (0..self.num_next as usize)
            .map(|i| {
                let nb = &self.next_balls[i];
                ((nb.row as usize, nb.col as usize), nb.color)
            })
            .collect()
    }

    // ── Ball spawning ─────────────────────────────────────────────

    fn generate_next_balls(&mut self) {
        let empty = get_empty_cells(&self.board);
        let n_empty = empty.len();
        if n_empty == 0 {
            self.num_next = 0;
            return;
        }
        let n = BALLS_PER_TURN.min(n_empty);
        let indices = self.rng.choice_no_replace(n_empty, n);
        let colors = self.rng.integers(1, NUM_COLORS as i64 + 1, n);
        for i in 0..n {
            self.next_balls[i] = NextBall {
                row: empty[indices[i]].0 as u8,
                col: empty[indices[i]].1 as u8,
                color: colors[i] as i8,
            };
        }
        self.num_next = n as u8;
    }

    fn spawn_balls(&mut self) {
        for i in 0..self.num_next as usize {
            let nb = self.next_balls[i];
            let (row, col) = (nb.row as usize, nb.col as usize);
            if self.board[row][col] == 0 {
                self.board[row][col] = nb.color;
            } else {
                let empty = get_empty_cells(&self.board);
                if !empty.is_empty() {
                    let idx = self.rng.randint(0, empty.len() as i64) as usize;
                    self.board[empty[idx].0][empty[idx].1] = nb.color;
                }
            }
        }
    }

    // ── Move execution ────────────────────────────────────────────

    /// Execute a move with full validation.
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

        self.execute_move(sr, sc, tr, tc)
    }

    /// Execute a move known to be legal (skip validation).
    #[inline]
    pub fn trusted_move(&mut self, sr: usize, sc: usize, tr: usize, tc: usize) {
        self.execute_move(sr, sc, tr, tc);
    }

    /// Shared move execution logic.
    #[inline]
    fn execute_move(
        &mut self,
        sr: usize,
        sc: usize,
        tr: usize,
        tc: usize,
    ) -> (bool, i32, usize, bool) {
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
            // Check spawned positions (snapshot next_balls before generate)
            let n = self.num_next as usize;
            let snapshot = self.next_balls;
            for i in 0..n {
                let (br, bc) = (snapshot[i].row as usize, snapshot[i].col as usize);
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
        assert_eq!(g.num_next, 0);
    }

    #[test]
    fn test_reset_places_balls() {
        let mut g = ColorLinesGame::new(42);
        g.reset();
        let n_balls = 81 - count_empty(&g.board);
        assert_eq!(n_balls, 3);
        assert_eq!(g.num_next, 3);
    }

    #[test]
    fn test_deterministic_reset() {
        let mut g1 = ColorLinesGame::new(42);
        g1.reset();
        let mut g2 = ColorLinesGame::new(42);
        g2.reset();
        assert_eq!(g1.board, g2.board);
        assert_eq!(g1.next_balls[..g1.num_next as usize], g2.next_balls[..g2.num_next as usize]);
    }

    #[test]
    fn test_clone_independent() {
        let mut g = ColorLinesGame::new(42);
        g.reset();
        let mut g2 = g.clone();
        g2.board[0][0] = 5;
        assert_ne!(g.board[0][0], g2.board[0][0]);
    }

    #[test]
    fn test_clone_is_cheap() {
        // Verify clone is truly stack-based (no heap)
        let mut g = ColorLinesGame::new(42);
        g.reset();
        let size = std::mem::size_of::<ColorLinesGame>();
        // Board(81) + next_balls(3×3=9) + num_next(1) + score(4) + turns(4)
        // + game_over(1) + rng(8) + padding ≈ ~120 bytes
        assert!(size < 200, "ColorLinesGame is {} bytes, expected <200", size);
    }

    #[test]
    fn test_next_balls_tuples_compat() {
        let mut g = ColorLinesGame::new(42);
        g.reset();
        let tuples = g.next_balls_tuples();
        assert_eq!(tuples.len(), g.num_next as usize);
        for (i, &((r, c), color)) in tuples.iter().enumerate() {
            assert_eq!(r, g.next_balls[i].row as usize);
            assert_eq!(c, g.next_balls[i].col as usize);
            assert_eq!(color, g.next_balls[i].color);
        }
    }
}
