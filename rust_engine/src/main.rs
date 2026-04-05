use colorlines98::board::*;
use colorlines98::game::ColorLinesGame;
use colorlines98::heuristic::get_best_move;
use colorlines98::tournament::tournament_player;
use rayon::prelude::*;
use serde::Serialize;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::time::Instant;

fn play_heuristic_game(seed: u64) -> (i32, i32) {
    let mut game = ColorLinesGame::new(seed);
    game.reset();
    while !game.game_over {
        match get_best_move(&game) {
            Some((sr, sc, tr, tc)) => { game.move_ball(sr, sc, tr, tc); }
            None => break,
        }
    }
    (game.score, game.turns)
}

// ── Data generation types ───────────────────────────────────────────

#[derive(Serialize)]
struct MoveRecord {
    board: [[i8; BOARD_SIZE]; BOARD_SIZE],
    next_balls: Vec<NextBallRecord>,
    num_next: usize,
    chosen_move: MoveRef,
    top_moves: Vec<MoveRef>,
    top_scores: Vec<f64>,
    num_top: usize,
    game_score: i32,
}

#[derive(Serialize)]
struct NextBallRecord {
    row: usize,
    col: usize,
    color: i8,
}

#[derive(Serialize)]
struct MoveRef {
    sr: usize,
    sc: usize,
    tr: usize,
    tc: usize,
}

#[derive(Serialize)]
struct GameRecord {
    seed: u64,
    score: i32,
    turns: i32,
    num_moves: usize,
    moves: Vec<MoveRecord>,
}

fn play_datagen_game(seed: u64, num_rollouts: usize, depth: usize, temp: f64) -> GameRecord {
    let mut game = ColorLinesGame::new(seed);
    game.reset();
    let mut records = Vec::with_capacity(512);
    let game_start = Instant::now();

    while !game.game_over {
        let board_snap = game.board;
        let next_snap: Vec<NextBallRecord> = (0..game.num_next as usize)
            .map(|i| NextBallRecord {
                row: game.next_balls[i].row as usize,
                col: game.next_balls[i].col as usize,
                color: game.next_balls[i].color,
            })
            .collect();
        let num_next = game.num_next as usize;

        match tournament_player(&mut game, num_rollouts, depth, temp) {
            Some(result) => {
                let (sr, sc, tr, tc) = result.chosen;
                let n_top = result.top_moves.len().min(5);
                records.push(MoveRecord {
                    board: board_snap,
                    next_balls: next_snap,
                    num_next,
                    chosen_move: MoveRef { sr, sc, tr, tc },
                    top_moves: result.top_moves[..n_top].iter()
                        .map(|&(m, _)| MoveRef { sr: m.0, sc: m.1, tr: m.2, tc: m.3 })
                        .collect(),
                    top_scores: result.top_moves[..n_top].iter().map(|&(_, s)| s).collect(),
                    num_top: n_top,
                    game_score: 0, // filled after game ends
                });
                game.move_ball(sr, sc, tr, tc);

                if game.turns % 500 == 0 {
                    let mps = game.turns as f64 / game_start.elapsed().as_secs_f64();
                    eprintln!("    [seed={}] turn={} score={} empty={} {:.0} mv/s",
                        seed, game.turns, game.score, count_empty(&game.board), mps);
                }
            }
            None => break,
        }
    }

    // Fill game_score for all moves
    for rec in records.iter_mut() {
        rec.game_score = game.score;
    }

    GameRecord {
        seed,
        score: game.score,
        turns: game.turns,
        num_moves: records.len(),
        moves: records,
    }
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
                let (score, _) = play_heuristic_game(i as u64);
                total_score.fetch_add(score as i64, Ordering::Relaxed);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 100 == 0 || done == n {
                    let elapsed = t0.elapsed().as_secs_f64();
                    let mean = total_score.load(Ordering::Relaxed) as f64 / done as f64;
                    eprintln!("  [{}/{}] mean={:.0} {:.0} games/s", done, n, mean, done as f64 / elapsed);
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
                let mut game = ColorLinesGame::new(seed);
                game.reset();
                while !game.game_over {
                    match tournament_player(&mut game, rollouts, 20, 3.23) {
                        Some(result) => {
                            game.move_ball(result.chosen.0, result.chosen.1, result.chosen.2, result.chosen.3);
                            if game.turns % 500 == 0 {
                                let mps = game.turns as f64 / game_start.elapsed().as_secs_f64();
                                eprintln!("    [seed={}] turn={} score={} empty={} {:.0} mv/s",
                                    seed, game.turns, game.score, count_empty(&game.board), mps);
                            }
                        }
                        None => break,
                    }
                }
                let game_time = game_start.elapsed().as_secs_f64();
                total_score.fetch_add(game.score as i64, Ordering::Relaxed);
                total_turns.fetch_add(game.turns as i64, Ordering::Relaxed);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                let elapsed = t0.elapsed().as_secs_f64();
                let mean = total_score.load(Ordering::Relaxed) as f64 / done as f64;
                let gph = done as f64 / elapsed * 3600.0;
                let eta_h = (n - done) as f64 / gph * 3600.0;
                eprintln!("  [{}/{}] seed={} score={} turns={} {:.0} mv/s | mean={:.0} {:.1} games/h ETA {:.1}h",
                    done, n, seed, game.score, game.turns, game.turns as f64 / game_time, mean, gph, eta_h / 3600.0);
            });

            let elapsed = t0.elapsed();
            let mean = total_score.load(Ordering::Relaxed) as f64 / n as f64;
            let avg_turns = total_turns.load(Ordering::Relaxed) as f64 / n as f64;
            println!("\nDone: {} games in {:.1}s ({:.1}h), mean={:.0}, avg_turns={:.0}",
                n, elapsed.as_secs_f64(), elapsed.as_secs_f64() / 3600.0, mean, avg_turns);
        }

        "datagen" => {
            let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            let rollouts: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
            let seed_start: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let out_dir = args.get(5).map(|s| s.as_str()).unwrap_or("data/expert_v2");
            let workers: usize = args.get(6).and_then(|s| s.parse().ok())
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
            rayon::ThreadPoolBuilder::new().num_threads(workers).build_global().ok();

            std::fs::create_dir_all(out_dir).unwrap();

            // Skip already completed seeds (resume-safe)
            let existing: std::collections::HashSet<u64> = std::fs::read_dir(out_dir)
                .unwrap()
                .filter_map(|e| {
                    let name = e.ok()?.file_name().into_string().ok()?;
                    if name.starts_with("game_seed") && name.ends_with(".json") {
                        name.trim_start_matches("game_seed")
                            .split('_').next()?
                            .parse().ok()
                    } else { None }
                })
                .collect();
            let seeds: Vec<u64> = (0..n as u64)
                .map(|i| seed_start + i)
                .filter(|s| !existing.contains(s))
                .collect();
            let n_skip = n - seeds.len();

            println!("=== Rust Tournament Data Generator ===");
            println!("Games: {}, Rollouts: {}, Depth: 20, Temp: 3.23", n, rollouts);
            println!("Seeds: {}..{}, Workers: {}, Output: {}", seed_start, seed_start + n as u64, workers, out_dir);
            if n_skip > 0 {
                println!("Skipping {} already completed games", n_skip);
            }
            println!();

            let total_games = seeds.len();
            let completed = AtomicUsize::new(0);
            let total_score = AtomicI64::new(0);
            let total_states = AtomicUsize::new(0);
            let t0 = Instant::now();

            seeds.into_par_iter().for_each(|seed| {
                let game_start = Instant::now();
                let record = play_datagen_game(seed, rollouts, 20, 3.23);
                let game_time = game_start.elapsed().as_secs_f64();
                let mps = record.turns as f64 / game_time;

                let fname = format!("{}/game_seed{}_score{}.json", out_dir, seed, record.score);
                std::fs::write(&fname, serde_json::to_string(&record).unwrap()).unwrap();

                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                total_score.fetch_add(record.score as i64, Ordering::Relaxed);
                total_states.fetch_add(record.num_moves, Ordering::Relaxed);

                let elapsed = t0.elapsed().as_secs_f64();
                let mean = total_score.load(Ordering::Relaxed) as f64 / done as f64;
                let gph = done as f64 / elapsed * 3600.0;
                let eta_h = (total_games - done) as f64 / gph * 3600.0;
                let states = total_states.load(Ordering::Relaxed);
                eprintln!("  [{}/{}] seed={} score={} turns={} {:.0} mv/s | mean={:.0} states={} {:.1} games/h ETA {:.1}h",
                    done + n_skip, n, seed, record.score, record.turns, mps, mean, states, gph, eta_h / 3600.0);
            });

            let elapsed = t0.elapsed();
            let mean = total_score.load(Ordering::Relaxed) as f64 / total_games.max(1) as f64;
            let states = total_states.load(Ordering::Relaxed);
            println!("\n=== Complete ===");
            println!("{} games in {:.1}s ({:.1}h), mean={:.0}, total_states={}",
                total_games, elapsed.as_secs_f64(), elapsed.as_secs_f64() / 3600.0, mean, states);
        }

        _ => {
            println!("Color Lines 98 — Rust Engine");
            println!();
            println!("Usage:");
            println!("  colorlines98 heuristic [n_games] [workers]");
            println!("  colorlines98 tournament [n_games] [rollouts] [seed_start] [workers]");
            println!("  colorlines98 datagen [n_games] [rollouts] [seed_start] [out_dir] [workers]");
        }
    }
}
