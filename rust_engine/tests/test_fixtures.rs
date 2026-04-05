/// Cross-language fixture tests.
/// Loads game/tests/fixtures.json and verifies Rust produces identical results.
use colorlines98::board::*;
use serde_json::Value;
use std::fs;

fn load_fixtures() -> Value {
    let data = fs::read_to_string("../game/tests/fixtures.json")
        .expect("Failed to load fixtures.json");
    serde_json::from_str(&data).expect("Failed to parse JSON")
}

fn json_to_board(val: &Value) -> Board {
    let mut board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
    let rows = val.as_array().unwrap();
    for (r, row) in rows.iter().enumerate() {
        for (c, cell) in row.as_array().unwrap().iter().enumerate() {
            // Handle both integer and float JSON values
            board[r][c] = if let Some(i) = cell.as_i64() {
                i as i8
            } else if let Some(f) = cell.as_f64() {
                f as i8
            } else {
                panic!("unexpected JSON value: {cell}");
            };
        }
    }
    board
}

// ── Score fixtures ──────────────────────────────────────────────────

#[test]
fn test_calculate_score_fixtures() {
    let fixtures = load_fixtures();
    for fix in fixtures["score"].as_array().unwrap() {
        let n = fix["input"]["num_balls"].as_u64().unwrap() as usize;
        let expected = fix["expected"].as_i64().unwrap() as i32;
        assert_eq!(
            calculate_score(n),
            expected,
            "calculate_score({n})"
        );
    }
}

// ── Component labeling fixtures ─────────────────────────────────────

#[test]
fn test_label_empty_components_fixtures() {
    let fixtures = load_fixtures();
    for (i, fix) in fixtures["components"].as_array().unwrap().iter().enumerate() {
        let board = json_to_board(&fix["input"]["board"]);
        let expected = json_to_board(&fix["expected"]["labels"]);
        let got = label_empty_components(&board);
        assert_eq!(
            got, expected,
            "label_empty_components fixture {i}"
        );
    }
}

// ── Line detection fixtures ─────────────────────────────────────────

#[test]
fn test_find_lines_at_fixtures() {
    let fixtures = load_fixtures();
    for fix in fixtures["lines"].as_array().unwrap() {
        if fix["function"].as_str().unwrap() != "find_lines_at" {
            continue;
        }
        let name = fix["name"].as_str().unwrap();
        let board = json_to_board(&fix["input"]["board"]);
        let row = fix["input"]["row"].as_u64().unwrap() as usize;
        let col = fix["input"]["col"].as_u64().unwrap() as usize;
        let expected = fix["expected"]["count"].as_i64().unwrap() as usize;
        let got = find_lines_at(&board, row, col);
        assert_eq!(got, expected, "find_lines_at '{name}'");
    }
}

#[test]
fn test_clear_lines_at_fixtures() {
    let fixtures = load_fixtures();
    for fix in fixtures["lines"].as_array().unwrap() {
        if fix["function"].as_str().unwrap() != "clear_lines_at" {
            continue;
        }
        let name = fix["name"].as_str().unwrap();
        let mut board = json_to_board(&fix["input"]["board"]);
        let row = fix["input"]["row"].as_u64().unwrap() as usize;
        let col = fix["input"]["col"].as_u64().unwrap() as usize;
        let expected_cleared = fix["expected"]["cleared"].as_i64().unwrap() as usize;
        let expected_board = json_to_board(&fix["expected"]["board_after"]);
        let got = clear_lines_at(&mut board, row, col);
        assert_eq!(got, expected_cleared, "clear_lines_at '{name}' count");
        assert_eq!(board, expected_board, "clear_lines_at '{name}' board");
    }
}

// ── Reachability fixtures ───────────────────────────────────────────

#[test]
fn test_is_reachable_fixtures() {
    let fixtures = load_fixtures();
    for fix in fixtures["reachability"].as_array().unwrap() {
        if fix["function"].as_str().unwrap() != "is_reachable" {
            continue;
        }
        let labels = json_to_board(&fix["input"]["labels"]);
        let sr = fix["input"]["sr"].as_u64().unwrap() as usize;
        let sc = fix["input"]["sc"].as_u64().unwrap() as usize;
        let tr = fix["input"]["tr"].as_u64().unwrap() as usize;
        let tc = fix["input"]["tc"].as_u64().unwrap() as usize;
        let expected = fix["expected"].as_bool().unwrap();
        let got = is_reachable(&labels, sr, sc, tr, tc);
        assert_eq!(got, expected, "is_reachable({sr},{sc})->({tr},{tc})");
    }
}

#[test]
fn test_get_target_mask_fixtures() {
    let fixtures = load_fixtures();
    for fix in fixtures["reachability"].as_array().unwrap() {
        if fix["function"].as_str().unwrap() != "get_target_mask" {
            continue;
        }
        let labels = json_to_board(&fix["input"]["labels"]);
        let sr = fix["input"]["sr"].as_u64().unwrap() as usize;
        let sc = fix["input"]["sc"].as_u64().unwrap() as usize;
        let expected_mask = json_to_board(&fix["expected"]["mask"]);
        let got = get_target_mask(&labels, sr, sc);
        // Compare as i8 (mask is 0/1 stored as i8 in fixture)
        assert_eq!(got, expected_mask, "get_target_mask({sr},{sc})");
    }
}
