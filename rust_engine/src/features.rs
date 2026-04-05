/// 30-feature spatial extractor + ML oracle scoring.
///
/// Features are extracted per-move for ML oracle blending.
/// Oracle weights learned via pairwise logistic regression on tournament data.

use crate::board::*;

const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

/// ML Oracle weights (pairwise logistic regression on 263 tournament games).
const ORACLE_WEIGHTS: [f64; 30] = [
    0.7645402595283389,
    2.396826140790059,
    -0.2114201949186696,
    2.481137677655506,
    2.020942991802538,
    -0.06789555695172257,
    -0.06041654907994659,
    0.08953055980197468,
    0.004709477063435807,
    -0.0826407199145921,
    0.0006675980188234034,
    0.08932807026847846,
    0.0,
    0.02082095348799286,
    0.09679350770319028,
    0.1318403558417833,
    -0.1096628463446392,
    0.2298056929583091,
    -0.0416024738382791,
    0.0,
    0.01834299615670471,
    0.09320812185813021,
    -0.5609906371714592,
    -0.006349524391907876,
    0.4668976485332657,
    1.56985815579482,
    0.2822469722104561,
    -0.6043081933651832,
    0.0,
    0.0,
];

/// Blend weight for ML oracle (5% of normalized score).
pub const ML_BLEND: f64 = 0.05;

#[inline]
fn line_len(board: &Board, r: usize, c: usize, color: i8, dr: i32, dc: i32) -> i32 {
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

/// Extract 30 spatial features for a move. Temporarily modifies board.
pub fn extract_features(
    board: &mut Board,
    sr: usize, sc: usize, tr: usize, tc: usize, color: i8,
    next_r: &[usize; 3], next_c: &[usize; 3], next_color: &[i8; 3], n_next: usize,
) -> [f64; 30] {
    let mut f = [0.0f64; 30];
    board[sr][sc] = 0;
    board[tr][tc] = color;

    // [0] max_line
    let mut max_line = 1i32;
    for &(dr, dc) in &DIRS {
        let l = line_len(board, tr, tc, color, dr, dc);
        if l > max_line { max_line = l; }
    }
    f[0] = max_line as f64;

    // [1] clears
    f[1] = if max_line >= MIN_LINE_LENGTH as i32 { 1.0 } else { 0.0 };

    // [2] partial_dirs (directions with line >= 3)
    let mut pd = 0;
    for &(dr, dc) in &DIRS {
        if line_len(board, tr, tc, color, dr, dc) >= 3 { pd += 1; }
    }
    f[2] = pd as f64;

    // [3] cross_dirs (directions with line >= 5)
    let mut cd = 0;
    for &(dr, dc) in &DIRS {
        if line_len(board, tr, tc, color, dr, dc) >= MIN_LINE_LENGTH as i32 { cd += 1; }
    }
    f[3] = cd as f64;

    // [4] has_line4
    f[4] = if max_line == 4 { 1.0 } else { 0.0 };

    // [5] extends (empty cells extending from same-color runs)
    let mut ext = 0i32;
    for &(dr, dc) in &DIRS {
        let (mut cr, mut cc) = (tr as i32 + dr, tc as i32 + dc);
        while cr >= 0 && cr < 9 && cc >= 0 && cc < 9 && board[cr as usize][cc as usize] == color { cr += dr; cc += dc; }
        while cr >= 0 && cr < 9 && cc >= 0 && cc < 9 && board[cr as usize][cc as usize] == 0 { ext += 1; cr += dr; cc += dc; }
        let (mut cr, mut cc) = (tr as i32 - dr, tc as i32 - dc);
        while cr >= 0 && cr < 9 && cc >= 0 && cc < 9 && board[cr as usize][cc as usize] == color { cr -= dr; cc -= dc; }
        while cr >= 0 && cr < 9 && cc >= 0 && cc < 9 && board[cr as usize][cc as usize] == 0 { ext += 1; cr -= dr; cc -= dc; }
    }
    f[5] = ext as f64;

    // [6-10] positional features
    let center = BOARD_SIZE as i32 / 2;
    f[6] = (tr as i32 - center).unsigned_abs() as f64 + (tc as i32 - center).unsigned_abs() as f64;
    f[7] = if tr == 0 || tr == 8 || tc == 0 || tc == 8 { 1.0 } else { 0.0 };
    f[8] = if (tr == 0 || tr == 8) && (tc == 0 || tc == 8) { 1.0 } else { 0.0 };
    f[9] = (sr as i32 - center).unsigned_abs() as f64 + (sc as i32 - center).unsigned_abs() as f64;
    f[10] = (tr as i32 - sr as i32).unsigned_abs() as f64 + (tc as i32 - sc as i32).unsigned_abs() as f64;

    // [11] local_empty (8 neighbors of target)
    let mut le = 0;
    for dr in -1i32..=1 { for dc in -1i32..=1 {
        if dr == 0 && dc == 0 { continue; }
        let (nr, nc) = (tr as i32 + dr, tc as i32 + dc);
        if nr >= 0 && nr < 9 && nc >= 0 && nc < 9 && board[nr as usize][nc as usize] == 0 { le += 1; }
    }}
    f[11] = le as f64;

    // [12] total_empty
    let mut te = 0;
    for r in 0..9 { for c in 0..9 { if board[r][c] == 0 { te += 1; } } }
    f[12] = te as f64;

    // [13] holes (adjacent empty cells that are fully surrounded)
    let mut holes = 0;
    for &(dr, dc) in &[(0i32,1i32),(0,-1),(1,0),(-1,0)] {
        let (nr, nc) = (tr as i32 + dr, tc as i32 + dc);
        if nr >= 0 && nr < 9 && nc >= 0 && nc < 9 && board[nr as usize][nc as usize] == 0 {
            let mut surr = true;
            for &(dr2, dc2) in &[(0i32,1i32),(0,-1),(1,0),(-1,0)] {
                let (nr2, nc2) = (nr + dr2, nc + dc2);
                if nr2 >= 0 && nr2 < 9 && nc2 >= 0 && nc2 < 9 && board[nr2 as usize][nc2 as usize] == 0 { surr = false; break; }
            }
            if surr { holes += 1; }
        }
    }
    f[13] = holes as f64;

    // [14-16] color neighbors (orthogonal same, orthogonal diff, diagonal same)
    let (mut so, mut do_, mut sd) = (0, 0, 0);
    for &(dr, dc) in &[(0i32,1i32),(0,-1),(1,0),(-1,0)] {
        let (nr, nc) = (tr as i32 + dr, tc as i32 + dc);
        if nr >= 0 && nr < 9 && nc >= 0 && nc < 9 {
            let v = board[nr as usize][nc as usize];
            if v == color { so += 1; } else if v != 0 { do_ += 1; }
        }
    }
    for &(dr, dc) in &[(-1i32,-1i32),(-1,1),(1,-1),(1,1)] {
        let (nr, nc) = (tr as i32 + dr, tc as i32 + dc);
        if nr >= 0 && nr < 9 && nc >= 0 && nc < 9 && board[nr as usize][nc as usize] == color { sd += 1; }
    }
    f[14] = so as f64;
    f[15] = do_ as f64;
    f[16] = sd as f64;

    // [17] nearest_same (Manhattan distance to nearest same-color ball)
    let mut min_d = 99i32;
    for r in 0..9 { for c in 0..9 {
        if board[r][c] == color && (r != tr || c != tc) {
            let d = (r as i32 - tr as i32).abs() + (c as i32 - tc as i32).abs();
            if d < min_d { min_d = d; }
        }
    }}
    f[17] = if min_d < 99 { min_d as f64 } else { 0.0 };

    // [18] same_count
    let mut sc_ = 0;
    for r in 0..9 { for c in 0..9 { if board[r][c] == color { sc_ += 1; } } }
    f[18] = sc_ as f64;

    // [19] n_colors (distinct colors on board)
    let mut cs = [false; 8];
    for r in 0..9 { for c in 0..9 { if board[r][c] > 0 { cs[board[r][c] as usize] = true; } } }
    f[19] = cs.iter().filter(|&&x| x).count() as f64;

    // [20-21] source features
    f[20] = if (sr as i32 - center).abs() <= 1 && (sc as i32 - center).abs() <= 1 { 1.0 } else { 0.0 };
    let mut src_cong = 0;
    for dr in -1i32..=1 { for dc in -1i32..=1 {
        if dr == 0 && dc == 0 { continue; }
        let (nr, nc) = (sr as i32 + dr, sc as i32 + dc);
        if nr >= 0 && nr < 9 && nc >= 0 && nc < 9 && board[nr as usize][nc as usize] > 0 { src_cong += 1; }
    }}
    f[21] = src_cong as f64;

    // [22] break_penalty (max line broken at source)
    let mut max_broken = 0i32;
    for &(dr, dc) in &DIRS { for &sign in &[1i32, -1] {
        let (nr, nc) = (sr as i32 + sign*dr, sc as i32 + sign*dc);
        if nr >= 0 && nr < 9 && nc >= 0 && nc < 9 && board[nr as usize][nc as usize] == color {
            let l = line_len(board, nr as usize, nc as usize, color, dr, dc);
            if l > max_broken { max_broken = l; }
        }
    }}
    f[22] = max_broken as f64;

    // [23] center_empty (empty cells in 3x3 center)
    let mut ce = 0;
    for r in 3..6 { for c in 3..6 { if board[r][c] == 0 { ce += 1; } } }
    f[23] = ce as f64;

    // [24-27] spawn features (only if not clearing)
    if f[1] == 0.0 {
        for i in 0..n_next {
            if tr == next_r[i] && tc == next_c[i] { f[24] = 1.0; break; }
        }
        for i in 0..n_next {
            let (nr, nc, ncol) = (next_r[i], next_c[i], next_color[i]);
            if board[nr][nc] == 0 {
                let mut ml = 0i32;
                for &(dr, dc) in &DIRS {
                    let l = line_len(board, nr, nc, ncol, dr, dc);
                    if l > ml { ml = l; }
                }
                if ml >= (MIN_LINE_LENGTH as i32 - 1) { f[25] += 1.0; }
            }
            if next_color[i] == color {
                let d = (next_r[i] as i32 - tr as i32).abs() + (next_c[i] as i32 - tc as i32).abs();
                if d <= 2 { f[26] += 1.0; }
            }
        }
    }
    let mut sv = 0;
    for i in 0..n_next { if board[next_r[i]][next_c[i]] == 0 { sv += 1; } }
    f[27] = sv as f64;

    // [28-29] density features
    f[28] = (81 - te) as f64 / 81.0;
    f[29] = 1.0 / (te as f64 + 1.0);

    // Restore board
    board[sr][sc] = color;
    board[tr][tc] = 0;
    f
}

/// Raw ML oracle score: features · weights.
pub fn ml_score(
    board: &mut Board,
    sr: usize, sc: usize, tr: usize, tc: usize, color: i8,
    next_r: &[usize; 3], next_c: &[usize; 3], next_color: &[i8; 3], n_next: usize,
) -> f64 {
    let feats = extract_features(board, sr, sc, tr, tc, color, next_r, next_c, next_color, n_next);
    feats.iter().zip(ORACLE_WEIGHTS.iter()).map(|(f, w)| f * w).sum()
}

/// Normalize heuristic + ML scores and blend (h_norm + ml_norm * ML_BLEND).
pub fn normalize_and_blend(h_scores: &[f64], ml_scores: &[f64]) -> Vec<f64> {
    let n = h_scores.len();
    if n == 0 { return vec![]; }

    let h_mean: f64 = h_scores.iter().sum::<f64>() / n as f64;
    let h_var: f64 = h_scores.iter().map(|x| (x - h_mean).powi(2)).sum::<f64>() / n as f64;
    let h_std = h_var.sqrt().max(1e-8);

    let m_mean: f64 = ml_scores.iter().sum::<f64>() / n as f64;
    let m_var: f64 = ml_scores.iter().map(|x| (x - m_mean).powi(2)).sum::<f64>() / n as f64;
    let m_std = m_var.sqrt().max(1e-8);

    (0..n).map(|i| {
        let h_norm = (h_scores[i] - h_mean) / h_std;
        let m_norm = (ml_scores[i] - m_mean) / m_std;
        h_norm + m_norm * ML_BLEND
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_feature_values() {
        // Verified against old Rust code's verification test
        let mut board: Board = [
            [1, 0, 0, 0, 0, 0, 0, 5, 6],
            [0, 3, 0, 6, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 1, 2, 6, 0, 2, 0, 0],
            [0, 5, 0, 0, 3, 0, 0, 0, 2],
            [6, 1, 0, 4, 4, 4, 4, 3, 7],
            [0, 0, 4, 0, 0, 2, 7, 5, 4],
            [0, 1, 0, 0, 3, 2, 0, 3, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 2],
        ];
        let next_r: [usize; 3] = [4, 0, 3];
        let next_c: [usize; 3] = [2, 2, 5];
        let next_color: [i8; 3] = [1, 4, 5];

        let feats = extract_features(&mut board, 0, 0, 0, 1, 1, &next_r, &next_c, &next_color, 3);

        let expected: [f64; 30] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 9.0,
            7.0, 1.0, 0.0, 8.0, 1.0, 4.0,
            49.0, 0.0, 0.0, 1.0, 0.0, 4.0,
            4.0, 7.0, 0.0, 2.0, 1.0, 3.0,
            0.0, 0.0, 0.0, 3.0,
            0.3950617283950617, 0.02,
        ];

        for i in 0..30 {
            assert!(
                (feats[i] - expected[i]).abs() < 0.001,
                "feature[{i}]: got {}, expected {}",
                feats[i], expected[i]
            );
        }
    }

    #[test]
    fn test_known_ml_score() {
        let mut board: Board = [
            [1, 0, 0, 0, 0, 0, 0, 5, 6],
            [0, 3, 0, 6, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 1, 2, 6, 0, 2, 0, 0],
            [0, 5, 0, 0, 3, 0, 0, 0, 2],
            [6, 1, 0, 4, 4, 4, 4, 3, 7],
            [0, 0, 4, 0, 0, 2, 7, 5, 4],
            [0, 1, 0, 0, 3, 2, 0, 3, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 2],
        ];
        let next_r: [usize; 3] = [4, 0, 3];
        let next_c: [usize; 3] = [2, 2, 5];
        let next_color: [i8; 3] = [1, 4, 5];

        let score = ml_score(&mut board, 0, 0, 0, 1, 1, &next_r, &next_c, &next_color, 3);
        assert!((score - (-1.804945)).abs() < 0.01, "ml_score={score}, expected -1.804945");
    }

    #[test]
    fn test_board_restored_after_features() {
        let mut board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        board[0][0] = 3;
        let original = board;
        let _ = extract_features(&mut board, 0, 0, 4, 4, 3, &[0;3], &[0;3], &[0;3], 0);
        assert_eq!(board, original);
    }

    #[test]
    fn test_normalize_and_blend() {
        let h = vec![10.0, 20.0, 30.0];
        let m = vec![100.0, 50.0, 75.0];
        let blended = normalize_and_blend(&h, &m);
        assert_eq!(blended.len(), 3);
        // Higher heuristic = higher blended (h_norm dominates at 5% blend)
        assert!(blended[2] > blended[0], "highest h should rank highest");
    }
}
