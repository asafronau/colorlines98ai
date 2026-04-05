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
    let source_mask = get_source_mask(&game.board);
    let labels = label_empty_components(&game.board);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_move = None;
    let mut board = game.board;

    for sr in 0..BOARD_SIZE {
        for sc in 0..BOARD_SIZE {
            if source_mask[sr][sc] == 0 {
                continue;
            }
            let color = board[sr][sc];
            let target_mask = get_target_mask(&labels, sr, sc);
            for tr in 0..BOARD_SIZE {
                for tc in 0..BOARD_SIZE {
                    if target_mask[tr][tc] == 0 {
                        continue;
                    }
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

/// Sample a move using softmax over heuristic scores.
pub fn get_softmax_move(
    game: &crate::game::ColorLinesGame,
    temperature: f64,
    rng: &mut crate::rng::SimpleRng,
) -> Option<(usize, usize, usize, usize)> {
    let source_mask = get_source_mask(&game.board);
    let labels = label_empty_components(&game.board);
    let mut moves = Vec::with_capacity(256);
    let mut scores = Vec::with_capacity(256);
    let mut board = game.board;

    for sr in 0..BOARD_SIZE {
        for sc in 0..BOARD_SIZE {
            if source_mask[sr][sc] == 0 {
                continue;
            }
            let color = board[sr][sc];
            let target_mask = get_target_mask(&labels, sr, sc);
            for tr in 0..BOARD_SIZE {
                for tc in 0..BOARD_SIZE {
                    if target_mask[tr][tc] == 0 {
                        continue;
                    }
                    let s = evaluate_move(&mut board, sr, sc, tr, tc, color);
                    moves.push((sr, sc, tr, tc));
                    scores.push(s);
                }
            }
        }
    }

    if moves.is_empty() {
        return None;
    }

    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = scores
        .iter()
        .map(|s| ((s - max_s) / temperature).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let r = rng.next_f64();
    let mut cumul = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumul += p;
        if r <= cumul {
            return Some(moves[i]);
        }
    }
    Some(*moves.last().unwrap())
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
