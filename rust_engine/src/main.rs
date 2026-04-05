use colorlines98::board::*;
use colorlines98::game::ColorLinesGame;
use colorlines98::heuristic::get_best_move;
use colorlines98::tournament::tournament_player;
use colorlines98::rng::SimpleRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::time::Instant;

fn play_heuristic_game(seed: u64) -> (i32, i32) {
    let mut game = ColorLinesGame::new(seed);
    game.reset();
    while !game.game_over {
        match get_best_move(&game) {
            Some((sr, sc, tr, tc)) => {
                game.move_ball(sr, sc, tr, tc);
            }
            None => break,
        }
    }
    (game.score, game.turns)
}

fn play_tournament_game(seed: u64, num_rollouts: usize, depth: usize, temp: f64) -> (i32, i32) {
    let mut game = ColorLinesGame::new(seed);
    game.reset();
    let game_start = Instant::now();
    while !game.game_over {
        match tournament_player(&mut game, num_rollouts, depth, temp) {
            Some(result) => {
                game.move_ball(
                    result.chosen.0, result.chosen.1,
                    result.chosen.2, result.chosen.3,
                );
                if game.turns % 500 == 0 {
                    let mps = game.turns as f64 / game_start.elapsed().as_secs_f64();
                    eprintln!(
                        "    [seed={}] turn={} score={} empty={} {:.0} mv/s",
                        seed, game.turns, game.score, count_empty(&game.board), mps
                    );
                }
            }
            None => break,
        }
    }
    (game.score, game.turns)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match mode {
        "heuristic" => {
            let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            let workers: usize = args.get(3).and_then(|s| s.parse().ok())
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

            rayon::ThreadPoolBuilder::new().num_threads(workers).build_global().ok();

            println!("Heuristic: {} games, {} workers", n, workers);
            let t0 = Instant::now();
            let completed = AtomicUsize::new(0);
            let total_score = AtomicI64::new(0);

            (0..n).into_par_iter().for_each(|i| {
                let (score, _turns) = play_heuristic_game(i as u64);
                total_score.fetch_add(score as i64, Ordering::Relaxed);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 100 == 0 || done == n {
                    let elapsed = t0.elapsed().as_secs_f64();
                    let mean = total_score.load(Ordering::Relaxed) as f64 / done as f64;
                    let gps = done as f64 / elapsed;
                    eprintln!("  [{}/{}] mean={:.0} {:.0} games/s", done, n, mean, gps);
                }
            });

            let elapsed = t0.elapsed();
            let mean = total_score.load(Ordering::Relaxed) as f64 / n as f64;
            println!("Done: {} games in {:.2}s ({:.0} games/s), mean={:.0}",
                n, elapsed.as_secs_f64(), n as f64 / elapsed.as_secs_f64(), mean);
        }

        "tournament" => {
            let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let rollouts: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
            let seed_start: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let workers: usize = args.get(5).and_then(|s| s.parse().ok())
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

            rayon::ThreadPoolBuilder::new().num_threads(workers).build_global().ok();

            println!("Tournament {}: {} games (seeds {}..{}), {} workers, depth=20, temp=3.23",
                rollouts, n, seed_start, seed_start + n as u64, workers);
            let t0 = Instant::now();
            let completed = AtomicUsize::new(0);
            let total_score = AtomicI64::new(0);
            let total_turns = AtomicI64::new(0);

            (0..n).into_par_iter().for_each(|i| {
                let seed = seed_start + i as u64;
                let game_start = Instant::now();
                let (score, turns) = play_tournament_game(seed, rollouts, 20, 3.23);
                let game_time = game_start.elapsed().as_secs_f64();

                total_score.fetch_add(score as i64, Ordering::Relaxed);
                total_turns.fetch_add(turns as i64, Ordering::Relaxed);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;

                let elapsed = t0.elapsed().as_secs_f64();
                let mean = total_score.load(Ordering::Relaxed) as f64 / done as f64;
                let gph = done as f64 / elapsed * 3600.0;
                let eta_h = (n - done) as f64 / gph * 3600.0;
                let mps = turns as f64 / game_time;
                eprintln!(
                    "  [{}/{}] seed={} score={} turns={} {:.0} mv/s | mean={:.0} {:.1} games/h ETA {:.1}h",
                    done, n, seed, score, turns, mps, mean, gph, eta_h / 3600.0
                );
            });

            let elapsed = t0.elapsed();
            let mean = total_score.load(Ordering::Relaxed) as f64 / n as f64;
            let avg_turns = total_turns.load(Ordering::Relaxed) as f64 / n as f64;
            println!("\nDone: {} games in {:.1}s ({:.1}h), mean={:.0}, avg_turns={:.0}",
                n, elapsed.as_secs_f64(), elapsed.as_secs_f64() / 3600.0, mean, avg_turns);
        }

        _ => {
            println!("Color Lines 98 — Rust Engine");
            println!();
            println!("Usage:");
            println!("  colorlines98 heuristic [n_games] [workers]");
            println!("  colorlines98 tournament [n_games] [rollouts] [seed_start] [workers]");
        }
    }
}
