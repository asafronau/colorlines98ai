/// Fast heuristic move evaluation — CMA-ES optimized 5 weights.
///
/// Evaluates moves by: (1) immediate line clears, (2) partial line potential,
/// (3) break penalty for splitting existing lines.

use crate::board::*;

/// CMA-ES optimized weights: [clear_mult, clear_base, partial_pow2, partial_linear, break_penalty]
const W: [f64; 5] = [14.6, 109.4, 5.7, 1.38, 2.4];
const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

#[inline]
fn line_length(board: &Board, r: usize, c: usize, color: i8, dr: i32, dc: i32) -> i32 {
    let mut length = 1i32;
    let (mut cr, mut cc) = (r as i32 + dr, c as i32 + dc);
    while cr >= 0 && cr < BOARD_SIZE as i32 && cc >= 0 && cc < BOARD_SIZE as i32
        && board[cr as usize][cc as usize] == color
    {
        length += 1;
        cr += dr;
        cc += dc;
    }
    let (mut cr, mut cc) = (r as i32 - dr, c as i32 - dc);
    while cr >= 0 && cr < BOARD_SIZE as i32 && cc >= 0 && cc < BOARD_SIZE as i32
        && board[cr as usize][cc as usize] == color
    {
        length += 1;
        cr -= dr;
        cc -= dc;
    }
    length
}

#[inline]
fn max_line_at(board: &Board, r: usize, c: usize, color: i8) -> i32 {
    DIRS.iter()
        .map(|&(dr, dc)| line_length(board, r, c, color, dr, dc))
        .max()
        .unwrap_or(1)
}

#[inline]
fn empty_extends(board: &Board, r: usize, c: usize, color: i8, dr: i32, dc: i32) -> i32 {
    let mut extends = 0;
    // Forward: skip same-color, count empty
    let (mut cr, mut cc) = (r as i32 + dr, c as i32 + dc);
    while cr >= 0 && cr < BOARD_SIZE as i32 && cc >= 0 && cc < BOARD_SIZE as i32
        && board[cr as usize][cc as usize] == color
    {
        cr += dr;
        cc += dc;
    }
    while cr >= 0 && cr < BOARD_SIZE as i32 && cc >= 0 && cc < BOARD_SIZE as i32
        && board[cr as usize][cc as usize] == 0
    {
        extends += 1;
        cr += dr;
        cc += dc;
    }
    // Backward
    let (mut cr, mut cc) = (r as i32 - dr, c as i32 - dc);
    while cr >= 0 && cr < BOARD_SIZE as i32 && cc >= 0 && cc < BOARD_SIZE as i32
        && board[cr as usize][cc as usize] == color
    {
        cr -= dr;
        cc -= dc;
    }
    while cr >= 0 && cr < BOARD_SIZE as i32 && cc >= 0 && cc < BOARD_SIZE as i32
        && board[cr as usize][cc as usize] == 0
    {
        extends += 1;
        cr -= dr;
        cc -= dc;
    }
    extends
}

/// Evaluate a move using CMA-ES heuristic weights.
/// Temporarily modifies board and restores it (no allocation).
pub fn evaluate_move(board: &mut Board, sr: usize, sc: usize, tr: usize, tc: usize, color: i8) -> f64 {
    board[sr][sc] = 0;
    board[tr][tc] = color;

    let mut score = 0.0f64;

    // 1) Clearable line
    let ml = max_line_at(board, tr, tc, color);
    if ml >= MIN_LINE_LENGTH as i32 {
        let pts = ml * (ml - 4);
        score += pts as f64 * W[0] + W[1];
    }

    // 2) Partial line potential
    for &(dr, dc) in &DIRS {
        let length = line_length(board, tr, tc, color, dr, dc);
        if length >= 2 {
            let ext = empty_extends(board, tr, tc, color, dr, dc);
            if length + ext >= MIN_LINE_LENGTH as i32 {
                score += ((length - 1) * (length - 1)) as f64 * W[2];
            } else {
                score += (length - 1) as f64 * W[3];
            }
        }
    }

    // 3) Break penalty — moving away from existing lines
    for &(dr, dc) in &DIRS {
        for &sign in &[1i32, -1] {
            let (nr, nc) = (sr as i32 + sign * dr, sc as i32 + sign * dc);
            if nr >= 0
                && nr < BOARD_SIZE as i32
                && nc >= 0
                && nc < BOARD_SIZE as i32
                && board[nr as usize][nc as usize] == color
            {
                let broken_len = line_length(board, nr as usize, nc as usize, color, dr, dc);
                let old_len = broken_len + 1;
                if old_len >= 3 {
                    score -= (old_len - 1) as f64 * W[4];
                }
            }
        }
    }

    // Restore
    board[sr][sc] = color;
    board[tr][tc] = 0;
    score
}

/// Find the best move using the heuristic.
pub fn get_best_move(game: &crate::game::ColorLinesGame) -> Option<(usize, usize, usize, usize)> {
    let mut buf = MoveBuffer::new();
    buf.fill(game);
    buf.best_move()
}

/// Fast rollout move: first check for immediate clears (very cheap),
/// then fall back to softmax over all moves if none found.
/// Returns the move and whether early-exit was used.
pub fn get_rollout_move(
    game: &crate::game::ColorLinesGame,
    temperature: f64,
    rng: &mut crate::rng::SimpleRng,
    buf: &mut MoveBuffer,
) -> Option<(usize, usize, usize, usize)> {
    let source_mask = get_source_mask(&game.board);
    let labels = label_empty_components(&game.board);
    let mut board = game.board;

    // Fast path: check for immediate line clears first (no full evaluation needed)
    for sr in 0..BOARD_SIZE {
        for sc in 0..BOARD_SIZE {
            if source_mask[sr][sc] == 0 { continue; }
            let color = board[sr][sc];
            // Check if this ball is part of a line of 4 (moving it could complete/break)
            let ml = max_line_at_fast(&board, sr, sc, color);
            if ml >= 4 {
                // This ball is promising — check its targets for clears
                let target_mask = get_target_mask(&labels, sr, sc);
                for tr in 0..BOARD_SIZE {
                    for tc in 0..BOARD_SIZE {
                        if target_mask[tr][tc] == 0 { continue; }
                        board[sr][sc] = 0;
                        board[tr][tc] = color;
                        let new_ml = max_line_at_fast(&board, tr, tc, color);
                        board[sr][sc] = color;
                        board[tr][tc] = 0;
                        if new_ml >= MIN_LINE_LENGTH as i32 {
                            return Some((sr, sc, tr, tc));
                        }
                    }
                }
            }
        }
    }

    // Slow path: full evaluation + softmax
    buf.fill(game);
    if temperature > 0.0 {
        buf.softmax_move(temperature, rng)
    } else {
        buf.best_move()
    }
}

#[inline]
fn max_line_at_fast(board: &Board, r: usize, c: usize, color: i8) -> i32 {
    DIRS.iter()
        .map(|&(dr, dc)| line_length(board, r, c, color, dr, dc))
        .max()
        .unwrap_or(1)
}

/// Reusable buffers for move evaluation (avoids allocation in hot loops).
pub struct MoveBuffer {
    pub moves: Vec<(usize, usize, usize, usize)>,
    pub scores: Vec<f64>,
}

impl MoveBuffer {
    pub fn new() -> Self {
        MoveBuffer {
            moves: Vec::with_capacity(1200),
            scores: Vec::with_capacity(1200),
        }
    }

    /// Collect all legal moves and their heuristic scores.
    pub fn fill(&mut self, game: &crate::game::ColorLinesGame) {
        self.fill_inner(&game.board, BOARD_SIZE); // no source limit
    }

    /// Fast fill: only evaluate moves from up to `max_sources` source balls.
    /// For rollouts where speed > accuracy.
    pub fn fill_fast(&mut self, game: &crate::game::ColorLinesGame, max_sources: usize) {
        self.fill_inner(&game.board, max_sources);
    }

    fn fill_inner(&mut self, board: &Board, max_sources: usize) {
        self.moves.clear();
        self.scores.clear();
        let source_mask = get_source_mask(board);
        let labels = label_empty_components(board);
        let mut board_mut = *board;
        let mut n_sources = 0;

        for sr in 0..BOARD_SIZE {
            for sc in 0..BOARD_SIZE {
                if source_mask[sr][sc] == 0 {
                    continue;
                }
                n_sources += 1;
                if n_sources > max_sources {
                    return;
                }
                let color = board_mut[sr][sc];
                let target_mask = get_target_mask(&labels, sr, sc);
                for tr in 0..BOARD_SIZE {
                    for tc in 0..BOARD_SIZE {
                        if target_mask[tr][tc] == 0 {
                            continue;
                        }
                        let s = evaluate_move(&mut board_mut, sr, sc, tr, tc, color);
                        self.moves.push((sr, sc, tr, tc));
                        self.scores.push(s);
                    }
                }
            }
        }
    }

    /// Pick the best move (greedy).
    pub fn best_move(&self) -> Option<(usize, usize, usize, usize)> {
        if self.moves.is_empty() {
            return None;
        }
        let best_idx = self.scores.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;
        Some(self.moves[best_idx])
    }

    /// Sample a move using softmax.
    pub fn softmax_move(&self, temperature: f64, rng: &mut crate::rng::SimpleRng) -> Option<(usize, usize, usize, usize)> {
        if self.moves.is_empty() {
            return None;
        }
        let max_s = self.scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut cumul = 0.0f64;
        let mut sum = 0.0f64;
        // Two-pass: compute sum, then sample
        for s in &self.scores {
            sum += ((s - max_s) / temperature).exp();
        }
        let r = rng.next_f64() * sum;
        for (i, s) in self.scores.iter().enumerate() {
            cumul += ((s - max_s) / temperature).exp();
            if r <= cumul {
                return Some(self.moves[i]);
            }
        }
        Some(*self.moves.last().unwrap())
    }
}

/// Sample a move using softmax over heuristic scores.
pub fn get_softmax_move(
    game: &crate::game::ColorLinesGame,
    temperature: f64,
    rng: &mut crate::rng::SimpleRng,
) -> Option<(usize, usize, usize, usize)> {
    let mut buf = MoveBuffer::new();
    buf.fill(game);
    buf.softmax_move(temperature, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_clear_5() {
        // 4 in a row + move completes to 5 → should score high
        let mut board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        board[4][0] = 2;
        board[4][1] = 2;
        board[4][2] = 2;
        board[4][3] = 2;
        // Move ball from somewhere to (4,4) to complete line of 5
        board[0][0] = 2;
        let score = evaluate_move(&mut board, 0, 0, 4, 4, 2);
        // Should get clear bonus: 5*(5-4)*W[0] + W[1] = 5*14.6 + 109.4 = 182.4
        assert!(score > 150.0, "clear-5 score={score}, expected >150");
    }

    #[test]
    fn test_evaluate_no_clear() {
        let mut board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        board[0][0] = 1;
        let score = evaluate_move(&mut board, 0, 0, 4, 4, 1);
        // No lines, no penalty → score near 0
        assert!(score.abs() < 50.0, "isolated move score={score}");
    }

    #[test]
    fn test_evaluate_restores_board() {
        let mut board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        board[0][0] = 3;
        let original = board;
        evaluate_move(&mut board, 0, 0, 4, 4, 3);
        assert_eq!(board, original);
    }

    #[test]
    fn test_get_best_move_returns_legal() {
        let mut game = crate::game::ColorLinesGame::new(42);
        game.reset();
        let mv = get_best_move(&game);
        assert!(mv.is_some());
        let (sr, sc, tr, tc) = mv.unwrap();
        assert!(game.board[sr][sc] != 0, "source must have a ball");
        assert!(game.board[tr][tc] == 0, "target must be empty");
    }

    #[test]
    fn test_heuristic_prefers_clear() {
        // Setup: line of 4, one move completes it, another moves elsewhere
        let mut board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        board[4][0] = 2;
        board[4][1] = 2;
        board[4][2] = 2;
        board[4][3] = 2;
        board[0][0] = 2; // can complete the line at (4,4) or move to (8,8)
        let score_clear = evaluate_move(&mut board, 0, 0, 4, 4, 2);
        let score_other = evaluate_move(&mut board, 0, 0, 8, 8, 2);
        assert!(
            score_clear > score_other,
            "clear={score_clear} should beat other={score_other}"
        );
    }

    #[test]
    fn test_known_heuristic_values() {
        // From old code's verification: specific board setup
        let mut board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        board[0][0] = 1;
        board[0][1] = 1;
        board[0][2] = 1;
        board[0][3] = 1;
        board[4][4] = 2;
        board[8][8] = 3;

        let s1 = evaluate_move(&mut board, 0, 0, 0, 4, 1);
        let s2 = evaluate_move(&mut board, 0, 3, 1, 3, 1);
        let s3 = evaluate_move(&mut board, 4, 4, 7, 7, 2);
        assert!(
            (s1 - 41.70).abs() < 0.01,
            "s1={s1}, expected 41.70"
        );
        assert!(
            (s2 - (-1.50)).abs() < 0.01,
            "s2={s2}, expected -1.50"
        );
        assert!(s3.abs() < 0.01, "s3={s3}, expected 0.00");
    }
}
