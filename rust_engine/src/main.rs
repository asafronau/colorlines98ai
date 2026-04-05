use colorlines98::board::*;
use colorlines98::game::ColorLinesGame;
use std::time::Instant;

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

fn main() {
    let n_games = 500;
    let t0 = Instant::now();
    let mut total_turns = 0u64;
    let mut total_score = 0i64;

    for seed in 0..n_games {
        let mut game = ColorLinesGame::new(seed);
        game.reset();
        while !game.game_over {
            match first_legal_move(&game) {
                Some((sr, sc, tr, tc)) => { game.move_ball(sr, sc, tr, tc); }
                None => break,
            }
        }
        total_turns += game.turns as u64;
        total_score += game.score as i64;
    }

    let elapsed = t0.elapsed();
    println!("Rust: {n_games} games in {:.2}s", elapsed.as_secs_f64());
    println!("  {:.0} games/s", n_games as f64 / elapsed.as_secs_f64());
    println!("  total turns: {total_turns}, mean score: {:.0}", total_score as f64 / n_games as f64);
    println!("  {:.1} us/turn", elapsed.as_micros() as f64 / total_turns as f64);
}
