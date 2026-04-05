/// Game replay fixture tests.
/// Verifies Rust game produces identical board states and scores as Python.
use colorlines98::board::*;
use colorlines98::game::{ColorLinesGame, NextBall};
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

#[test]
fn test_game_replay_board_after_reset() {
    let fixtures = load_fixtures();
    for fix in fixtures["game_replay"].as_array().unwrap() {
        let seed = fix["seed"].as_u64().unwrap();
        let expected_board = json_to_board(&fix["board_after_reset"]);

        let mut game = ColorLinesGame::new(seed);
        game.reset();

        assert_eq!(
            game.board, expected_board,
            "seed={seed}: board after reset mismatch"
        );
    }
}

#[test]
fn test_game_replay_next_balls_after_reset() {
    let fixtures = load_fixtures();
    for fix in fixtures["game_replay"].as_array().unwrap() {
        let seed = fix["seed"].as_u64().unwrap();
        let expected_nb = fix["next_balls_after_reset"].as_array().unwrap();

        let mut game = ColorLinesGame::new(seed);
        game.reset();

        assert_eq!(
            game.num_next as usize,
            expected_nb.len(),
            "seed={seed}: next_balls count mismatch"
        );

        for (i, nb) in expected_nb.iter().enumerate() {
            let row = nb["row"].as_u64().unwrap() as u8;
            let col = nb["col"].as_u64().unwrap() as u8;
            let color = nb["color"].as_i64().unwrap() as i8;
            assert_eq!(
                game.next_balls[i],
                NextBall { row, col, color },
                "seed={seed}: next_ball[{i}] mismatch"
            );
        }
    }
}

#[test]
fn test_game_replay_moves() {
    let fixtures = load_fixtures();
    for fix in fixtures["game_replay"].as_array().unwrap() {
        let seed = fix["seed"].as_u64().unwrap();
        let moves = fix["moves"].as_array().unwrap();

        let mut game = ColorLinesGame::new(seed);
        game.reset();

        for (mi, mv) in moves.iter().enumerate() {
            let sr = mv["source"][0].as_u64().unwrap() as usize;
            let sc = mv["source"][1].as_u64().unwrap() as usize;
            let tr = mv["target"][0].as_u64().unwrap() as usize;
            let tc = mv["target"][1].as_u64().unwrap() as usize;
            let expected_valid = mv["valid"].as_bool().unwrap();
            let expected_cleared = mv["cleared"].as_u64().unwrap() as usize;
            let expected_score = mv["score_after"].as_i64().unwrap() as i32;
            let expected_board = json_to_board(&mv["board_after"]);
            let expected_game_over = mv["game_over"].as_bool().unwrap();

            let (valid, _, cleared, _) = game.move_ball(sr, sc, tr, tc);

            assert_eq!(
                valid, expected_valid,
                "seed={seed} move {mi}: valid mismatch"
            );
            assert_eq!(
                cleared, expected_cleared,
                "seed={seed} move {mi}: cleared mismatch"
            );
            assert_eq!(
                game.score, expected_score,
                "seed={seed} move {mi}: score mismatch"
            );
            assert_eq!(
                game.board, expected_board,
                "seed={seed} move {mi}: board mismatch"
            );
            assert_eq!(
                game.game_over, expected_game_over,
                "seed={seed} move {mi}: game_over mismatch"
            );
        }

        // Verify final state
        let expected_final_score = fix["final_score"].as_i64().unwrap() as i32;
        let expected_final_turns = fix["final_turns"].as_i64().unwrap() as i32;
        assert_eq!(
            game.score, expected_final_score,
            "seed={seed}: final score mismatch"
        );
        assert_eq!(
            game.turns, expected_final_turns,
            "seed={seed}: final turns mismatch"
        );
    }
}
