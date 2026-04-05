/// Board operations for Color Lines 98.
/// 9x9 board, 7 colors, lines of 5+ to clear.

pub const BOARD_SIZE: usize = 9;
pub const NUM_COLORS: i8 = 7;
pub const BALLS_PER_TURN: usize = 3;
pub const MIN_LINE_LENGTH: usize = 5;

pub type Board = [[i8; BOARD_SIZE]; BOARD_SIZE];

/// Score formula: n * (n - 4) for n >= 5, else 0.
pub fn calculate_score(num_balls: usize) -> i32 {
    if num_balls < MIN_LINE_LENGTH {
        0
    } else {
        (num_balls * (num_balls - 4)) as i32
    }
}

/// Count empty cells on board.
pub fn count_empty(board: &Board) -> usize {
    let mut n = 0;
    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            if board[r][c] == 0 {
                n += 1;
            }
        }
    }
    n
}

/// Return vec of (row, col) for all empty cells, in row-major order.
pub fn get_empty_cells(board: &Board) -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            if board[r][c] == 0 {
                cells.push((r, c));
            }
        }
    }
    cells
}

/// BFS-label connected components of empty cells.
/// Returns (9,9) array: 0 = ball, 1+ = component ID.
pub fn label_empty_components(board: &Board) -> Board {
    let mut labels = [[0i8; BOARD_SIZE]; BOARD_SIZE];
    let mut queue_r = [0u8; 81];
    let mut queue_c = [0u8; 81];
    let mut current: i8 = 0;

    for sr in 0..BOARD_SIZE {
        for sc in 0..BOARD_SIZE {
            if board[sr][sc] != 0 || labels[sr][sc] != 0 {
                continue;
            }
            current += 1;
            labels[sr][sc] = current;
            queue_r[0] = sr as u8;
            queue_c[0] = sc as u8;
            let mut head = 0;
            let mut tail = 1;

            while head < tail {
                let r = queue_r[head] as i32;
                let c = queue_c[head] as i32;
                head += 1;

                for &(dr, dc) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    let nr = r + dr;
                    let nc = c + dc;
                    if nr >= 0
                        && nr < BOARD_SIZE as i32
                        && nc >= 0
                        && nc < BOARD_SIZE as i32
                    {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        if board[nr][nc] == 0 && labels[nr][nc] == 0 {
                            labels[nr][nc] = current;
                            queue_r[tail] = nr as u8;
                            queue_c[tail] = nc as u8;
                            tail += 1;
                        }
                    }
                }
            }
        }
    }
    labels
}

/// Find total balls in lines of 5+ through (row, col). Does NOT clear.
pub fn find_lines_at(board: &Board, row: usize, col: usize) -> usize {
    let color = board[row][col];
    if color == 0 {
        return 0;
    }
    let mut board_copy = *board;
    collect_line_cells(&mut board_copy, row, col, color, false)
}

/// Clear lines of 5+ through (row, col). Mutates board. Returns count cleared.
pub fn clear_lines_at(board: &mut Board, row: usize, col: usize) -> usize {
    let color = board[row][col];
    if color == 0 {
        return 0;
    }
    collect_line_cells(board, row, col, color, true)
}

/// Shared implementation for find_lines_at / clear_lines_at.
fn collect_line_cells(
    board: &mut Board,
    row: usize,
    col: usize,
    color: i8,
    do_clear: bool,
) -> usize {
    let mut clear_r = [0u8; 36];
    let mut clear_c = [0u8; 36];
    let mut n_clear = 0;

    let dirs: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

    for &(dr, dc) in &dirs {
        let mut line_r = [0u8; 9];
        let mut line_c = [0u8; 9];
        line_r[0] = row as u8;
        line_c[0] = col as u8;
        let mut n = 1usize;

        // Positive direction
        let mut r = row as i32 + dr;
        let mut c = col as i32 + dc;
        while r >= 0 && r < BOARD_SIZE as i32 && c >= 0 && c < BOARD_SIZE as i32 {
            if board[r as usize][c as usize] != color {
                break;
            }
            line_r[n] = r as u8;
            line_c[n] = c as u8;
            n += 1;
            r += dr;
            c += dc;
        }

        // Negative direction
        r = row as i32 - dr;
        c = col as i32 - dc;
        while r >= 0 && r < BOARD_SIZE as i32 && c >= 0 && c < BOARD_SIZE as i32 {
            if board[r as usize][c as usize] != color {
                break;
            }
            line_r[n] = r as u8;
            line_c[n] = c as u8;
            n += 1;
            r -= dr;
            c -= dc;
        }

        if n >= MIN_LINE_LENGTH {
            for i in 0..n {
                // Deduplicate
                let mut already = false;
                for j in 0..n_clear {
                    if clear_r[j] == line_r[i] && clear_c[j] == line_c[i] {
                        already = true;
                        break;
                    }
                }
                if !already {
                    clear_r[n_clear] = line_r[i];
                    clear_c[n_clear] = line_c[i];
                    n_clear += 1;
                }
            }
        }
    }

    if do_clear {
        for i in 0..n_clear {
            board[clear_r[i] as usize][clear_c[i] as usize] = 0;
        }
    }

    n_clear
}

/// Check if target is reachable from source via empty cells.
pub fn is_reachable(labels: &Board, sr: usize, sc: usize, tr: usize, tc: usize) -> bool {
    if labels[tr][tc] == 0 {
        return false;
    }
    let target_label = labels[tr][tc];
    for &(dr, dc) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
        let nr = sr as i32 + dr;
        let nc = sc as i32 + dc;
        if nr >= 0 && nr < BOARD_SIZE as i32 && nc >= 0 && nc < BOARD_SIZE as i32 {
            if labels[nr as usize][nc as usize] == target_label {
                return true;
            }
        }
    }
    false
}

/// Return (9,9) mask: 1 for empty cells reachable from source, 0 otherwise.
/// Checks ALL adjacent components (a ball can reach any component it touches).
pub fn get_target_mask(labels: &Board, sr: usize, sc: usize) -> Board {
    let mut mask = [[0i8; BOARD_SIZE]; BOARD_SIZE];
    let mut reachable_labels = [0i8; 4];
    let mut n_labels = 0;

    for &(dr, dc) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
        let nr = sr as i32 + dr;
        let nc = sc as i32 + dc;
        if nr >= 0 && nr < BOARD_SIZE as i32 && nc >= 0 && nc < BOARD_SIZE as i32 {
            let lbl = labels[nr as usize][nc as usize];
            if lbl > 0 {
                let mut found = false;
                for i in 0..n_labels {
                    if reachable_labels[i] == lbl {
                        found = true;
                        break;
                    }
                }
                if !found {
                    reachable_labels[n_labels] = lbl;
                    n_labels += 1;
                }
            }
        }
    }

    if n_labels == 0 {
        return mask;
    }

    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            if labels[r][c] > 0 {
                for i in 0..n_labels {
                    if labels[r][c] == reachable_labels[i] {
                        mask[r][c] = 1;
                        break;
                    }
                }
            }
        }
    }
    mask
}

/// Return (9,9) mask: 1 where a ball has ≥1 adjacent empty cell.
pub fn get_source_mask(board: &Board) -> Board {
    let mut mask = [[0i8; BOARD_SIZE]; BOARD_SIZE];
    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            if board[r][c] == 0 {
                continue;
            }
            for &(dr, dc) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0
                    && nr < BOARD_SIZE as i32
                    && nc >= 0
                    && nc < BOARD_SIZE as i32
                    && board[nr as usize][nc as usize] == 0
                {
                    mask[r][c] = 1;
                    break;
                }
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_below_5() {
        for n in 0..5 {
            assert_eq!(calculate_score(n), 0);
        }
    }

    #[test]
    fn test_score_5() {
        assert_eq!(calculate_score(5), 5);
    }

    #[test]
    fn test_score_6() {
        assert_eq!(calculate_score(6), 12);
    }

    #[test]
    fn test_score_10() {
        assert_eq!(calculate_score(10), 60);
    }

    #[test]
    fn test_count_empty_full_board() {
        let board = [[1i8; BOARD_SIZE]; BOARD_SIZE];
        assert_eq!(count_empty(&board), 0);
    }

    #[test]
    fn test_count_empty_empty_board() {
        let board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        assert_eq!(count_empty(&board), 81);
    }

    #[test]
    fn test_empty_cells_order() {
        let board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        let cells = get_empty_cells(&board);
        assert_eq!(cells.len(), 81);
        assert_eq!(cells[0], (0, 0));
        assert_eq!(cells[80], (8, 8));
    }
}
