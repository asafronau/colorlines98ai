/// Full game parity test.
/// Plays 50 seeds to completion using first-legal-move strategy.
/// Verifies Rust final scores and turn counts match Python exactly.
use colorlines98::board::*;
use colorlines98::game::ColorLinesGame;
use serde_json::Value;
use std::fs;

fn load_reference() -> Value {
    let data = fs::read_to_string("../game/tests/full_game_reference.json")
        .expect("Failed to load full_game_reference.json");
    serde_json::from_str(&data).expect("Failed to parse JSON")
}

/// Find first legal move (same strategy as Python test).
fn first_legal_move(game: &ColorLinesGame) -> Option<(usize, usize, usize, usize)> {
    let source_mask = get_source_mask(&game.board);
    let labels = label_empty_components(&game.board);

    for sr in 0..BOARD_SIZE {
        for sc in 0..BOARD_SIZE {
            if source_mask[sr][sc] == 0 {
                continue;
            }
            let target_mask = get_target_mask(&labels, sr, sc);
            for tr in 0..BOARD_SIZE {
                for tc in 0..BOARD_SIZE {
                    if target_mask[tr][tc] > 0 {
                        return Some((sr, sc, tr, tc));
                    }
                }
            }
        }
    }
    None
}

#[test]
fn test_50_seeds_match_python() {
    let refs = load_reference();

    for seed in 0u64..50 {
        let mut game = ColorLinesGame::new(seed);
        game.reset();

        while !game.game_over {
            match first_legal_move(&game) {
                Some((sr, sc, tr, tc)) => {
                    game.move_ball(sr, sc, tr, tc);
                }
                None => break,
            }
        }

        let seed_str = seed.to_string();
        let expected_score = refs[&seed_str]["score"].as_i64().unwrap() as i32;
        let expected_turns = refs[&seed_str]["turns"].as_i64().unwrap() as i32;

        assert_eq!(
            game.score, expected_score,
            "seed={seed}: score mismatch (got {}, expected {expected_score})",
            game.score
        );
        assert_eq!(
            game.turns, expected_turns,
            "seed={seed}: turns mismatch (got {}, expected {expected_turns})",
            game.turns
        );
    }
}
