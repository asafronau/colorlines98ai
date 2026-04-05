/// Profile tournament player: measure time spent in each phase.

use colorlines98::board::*;
use colorlines98::game::ColorLinesGame;
use colorlines98::heuristic::{evaluate_move, get_best_move, get_softmax_move};
use colorlines98::tournament::tournament_player;
use colorlines98::rng::SimpleRng;
use std::time::Instant;

fn main() {
    let mut game = ColorLinesGame::new(42);
    game.reset();

    // Play 10 heuristic moves to get to mid-game
    for _ in 0..10 {
        if game.game_over { break; }
        if let Some((sr, sc, tr, tc)) = get_best_move(&game) {
            game.move_ball(sr, sc, tr, tc);
        }
    }

    let n_legal = {
        let sm = get_source_mask(&game.board);
        let labels = label_empty_components(&game.board);
        let mut count = 0;
        for sr in 0..BOARD_SIZE {
            for sc in 0..BOARD_SIZE {
                if sm[sr][sc] == 0 { continue; }
                let tm = get_target_mask(&labels, sr, sc);
                for tr in 0..BOARD_SIZE {
                    for tc in 0..BOARD_SIZE {
                        if tm[tr][tc] != 0 { count += 1; }
                    }
                }
            }
        }
        count
    };
    println!("Board: {} balls, {} empty, {} legal moves",
        81 - count_empty(&game.board), count_empty(&game.board), n_legal);
    println!("Game struct size: {} bytes", std::mem::size_of::<ColorLinesGame>());

    // ── Micro-benchmarks ──
    let mut board_copy = game.board;
    let color = game.board[0][0];

    let n = 100_000;
    let t0 = Instant::now();
    for _ in 0..n { evaluate_move(&mut board_copy, 0, 0, 4, 4, if color != 0 { color } else { 1 }); }
    println!("\nevaluate_move: {:.3}µs/call ({n} calls)", t0.elapsed().as_secs_f64() * 1e6 / n as f64);

    let n = 10_000;
    let t0 = Instant::now();
    for _ in 0..n { let _ = get_best_move(&game); }
    println!("get_best_move: {:.1}µs/call ({} legal moves, {n} calls)",
        t0.elapsed().as_secs_f64() * 1e6 / n as f64, n_legal);

    let n = 100_000;
    let t0 = Instant::now();
    for _ in 0..n { let _ = game.clone(); }
    println!("game.clone: {:.3}µs/call ({n} calls)", t0.elapsed().as_secs_f64() * 1e6 / n as f64);

    let n = 100_000;
    let t0 = Instant::now();
    for _ in 0..n { let _ = label_empty_components(&game.board); }
    println!("label_components: {:.3}µs/call ({n} calls)", t0.elapsed().as_secs_f64() * 1e6 / n as f64);

    let n = 100_000;
    let t0 = Instant::now();
    for _ in 0..n { let _ = get_source_mask(&game.board); }
    println!("get_source_mask: {:.3}µs/call ({n} calls)", t0.elapsed().as_secs_f64() * 1e6 / n as f64);

    let n = 100_000;
    let t0 = Instant::now();
    let labels = label_empty_components(&game.board);
    for _ in 0..n { let _ = get_target_mask(&labels, 0, 0); }
    println!("get_target_mask: {:.3}µs/call ({n} calls)", t0.elapsed().as_secs_f64() * 1e6 / n as f64);

    // ── get_best_move breakdown ──
    println!("\n--- get_best_move breakdown (1 call) ---");
    let t0 = Instant::now();
    let source_mask = get_source_mask(&game.board);
    let t_sm = t0.elapsed();

    let t0 = Instant::now();
    let labels = label_empty_components(&game.board);
    let t_cc = t0.elapsed();

    let mut n_sources = 0;
    let t0 = Instant::now();
    for sr in 0..BOARD_SIZE {
        for sc in 0..BOARD_SIZE {
            if source_mask[sr][sc] == 0 { continue; }
            n_sources += 1;
            let _ = get_target_mask(&labels, sr, sc);
        }
    }
    let t_tm = t0.elapsed();

    let t0 = Instant::now();
    let mut n_evals = 0;
    for sr in 0..BOARD_SIZE {
        for sc in 0..BOARD_SIZE {
            if source_mask[sr][sc] == 0 { continue; }
            let c = game.board[sr][sc];
            let tm = get_target_mask(&labels, sr, sc);
            for tr in 0..BOARD_SIZE {
                for tc in 0..BOARD_SIZE {
                    if tm[tr][tc] == 0 { continue; }
                    evaluate_move(&mut board_copy, sr, sc, tr, tc, c);
                    n_evals += 1;
                }
            }
        }
    }
    let t_ev = t0.elapsed();

    println!("  source_mask: {:.1}µs", t_sm.as_secs_f64() * 1e6);
    println!("  label_components: {:.1}µs", t_cc.as_secs_f64() * 1e6);
    println!("  target_masks ({n_sources} sources): {:.1}µs", t_tm.as_secs_f64() * 1e6);
    println!("  evaluate ({n_evals} moves): {:.1}µs ({:.3}µs/eval)",
        t_ev.as_secs_f64() * 1e6, t_ev.as_secs_f64() * 1e6 / n_evals as f64);

    // ── Single rollout profile ──
    println!("\n--- Single rollout (depth=20) ---");
    let mut rng = SimpleRng::new(99);
    let t0 = Instant::now();
    let n_rollouts = 100;
    for _ in 0..n_rollouts {
        let mut clone = game.clone_with_rng(SimpleRng::new(rng.next_u64()));
        for _ in 0..20 {
            if clone.game_over { break; }
            match get_softmax_move(&clone, 3.23, &mut SimpleRng::new(rng.next_u64())) {
                Some((sr, sc, tr, tc)) => { clone.move_ball(sr, sc, tr, tc); }
                None => break,
            }
        }
    }
    let t_roll = t0.elapsed();
    println!("  {n_rollouts} rollouts: {:.1}ms ({:.1}µs/rollout, {:.1}µs/step)",
        t_roll.as_secs_f64() * 1e3,
        t_roll.as_secs_f64() * 1e6 / n_rollouts as f64,
        t_roll.as_secs_f64() * 1e6 / n_rollouts as f64 / 20.0);

    // ── Full tournament move ──
    println!("\n--- Full tournament move (50 rollouts) ---");
    let mut rng = SimpleRng::new(99);
    let t0 = Instant::now();
    let _ = tournament_player(&mut game, 50, 20, 3.23);
    let t_tour = t0.elapsed();
    println!("  {:.1}ms/move ({:.1} mv/s)", t_tour.as_secs_f64() * 1e3, 1.0 / t_tour.as_secs_f64());

    println!("\n--- Full tournament move (200 rollouts) ---");
    let mut rng = SimpleRng::new(99);
    let t0 = Instant::now();
    let _ = tournament_player(&mut game, 200, 20, 3.23);
    let t_tour200 = t0.elapsed();
    println!("  {:.1}ms/move ({:.1} mv/s)", t_tour200.as_secs_f64() * 1e3, 1.0 / t_tour200.as_secs_f64());

    println!("\nTarget: 25 mv/s = 40ms/move");
}
