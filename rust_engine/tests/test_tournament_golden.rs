/// Golden test: tournament scores must be exactly reproducible.
/// Any optimization that changes these scores has altered the algorithm.

use colorlines98::game::ColorLinesGame;
use colorlines98::tournament::tournament_player;

fn play_tournament_game(seed: u64, rollouts: usize) -> (i32, i32) {
    let mut game = ColorLinesGame::new(seed);
    game.reset();
    while !game.game_over {
        match tournament_player(&mut game, rollouts, 20, 3.23) {
            Some(result) => {
                game.move_ball(
                    result.chosen.0, result.chosen.1,
                    result.chosen.2, result.chosen.3,
                );
            }
            None => break,
        }
    }
    (game.score, game.turns)
}

/// Golden scores: 50 rollouts, SplitMix64, depth=20, temp=3.23.
/// These MUST NOT change across optimizations.
const GOLDEN: [(u64, i32, i32); 10] = [
    (0, 1614, 756),
    (1, 4535, 2084),
    (2, 3768, 1786),
    (3, 1019, 460),
    (4, 3218, 1452),
    (5, 766, 377),
    (6, 2235, 1044),
    (7, 4433, 2065),
    (8, 5054, 2366),
    (9, 1007, 484),
];

#[test]
fn test_golden_seed_3() {
    // Seed 3 is the shortest game (460 turns) — fast to verify
    let (score, turns) = play_tournament_game(3, 50);
    assert_eq!((score, turns), (1019, 460),
        "seed=3: got ({score}, {turns}), expected (1019, 460)");
}

#[test]
fn test_golden_seed_5() {
    let (score, turns) = play_tournament_game(5, 50);
    assert_eq!((score, turns), (766, 377),
        "seed=5: got ({score}, {turns}), expected (766, 377)");
}

#[test]
#[ignore] // Run with: cargo test -- --ignored (slow: ~13 min for all 10)
fn test_all_golden_scores() {
    for &(seed, expected_score, expected_turns) in &GOLDEN {
        let (score, turns) = play_tournament_game(seed, 50);
        assert_eq!(
            (score, turns), (expected_score, expected_turns),
            "seed={seed}: got ({score}, {turns}), expected ({expected_score}, {expected_turns})"
        );
    }
}
