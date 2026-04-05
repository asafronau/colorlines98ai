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

    // ── 2-ply simulation (the bottleneck) ──
    println!("\n--- 2-ply simulation (matching tournament logic) ---");
    {
        game.ensure_cc();
        let src_bits = get_source_mask_bits(&game.board);
        let labels = game.cc_labels;
        let comp = ComponentMasks::from_labels(&labels);

        let mut ply = game.clone();
        let mut n_candidates = 0;
        let mut t_move = 0u64;
        let mut t_gbm = 0u64;
        let mut t_comp = 0u64;

        let t0 = Instant::now();
        let mut src = src_bits;
        while src != 0 {
            let si = src.trailing_zeros();
            src &= src - 1;
            let (sr, sc) = idx_to_rc(si);
            let color = game.board[sr][sc];
            let tgt = comp.target_mask(&labels, sr, sc);
            let mut t = tgt;
            while t != 0 {
                let ti = t.trailing_zeros();
                t &= t - 1;
                let (tr, tc) = idx_to_rc(ti);

                // Simulate 2-ply (without RNG to avoid side effects)
                ply.board = game.board;
                ply.next_balls = game.next_balls;
                ply.num_next = game.num_next;
                ply.score = game.score;
                ply.turns = game.turns;
                ply.game_over = game.game_over;
                ply.cc_labels = game.cc_labels;
                ply.cc_valid = true;

                let tm = Instant::now();
                ply.trusted_move(sr, sc, tr, tc);
                t_move += tm.elapsed().as_nanos() as u64;

                if !ply.game_over {
                    let tc2 = Instant::now();
                    ply.ensure_cc();
                    let ply_labels = ply.cc_labels;
                    let ply_comp = ComponentMasks::from_labels(&ply_labels);
                    t_comp += tc2.elapsed().as_nanos() as u64;

                    let tg = Instant::now();
                    let ply_src = get_source_mask_bits(&ply.board);
                    let mut best_score = f64::NEG_INFINITY;
                    let mut ps = ply_src;
                    while ps != 0 {
                        let psi = ps.trailing_zeros();
                        ps &= ps - 1;
                        let (psr, psc) = idx_to_rc(psi);
                        let pc = ply.board[psr][psc];
                        let pt = ply_comp.target_mask(&ply_labels, psr, psc);
                        let mut ptt = pt;
                        while ptt != 0 {
                            let pti = ptt.trailing_zeros();
                            ptt &= ptt - 1;
                            let (ptr, ptc) = idx_to_rc(pti);
                            let s = evaluate_move(&mut ply.board, psr, psc, ptr, ptc, pc);
                            if s > best_score { best_score = s; }
                        }
                    }
                    t_gbm += tg.elapsed().as_nanos() as u64;
                }
                n_candidates += 1;
            }
        }
        let total = t0.elapsed();
        println!("  {n_candidates} candidates in {:.1}ms", total.as_secs_f64() * 1e3);
        println!("  trusted_move: {:.1}ms ({:.1}µs/call)", t_move as f64 / 1e6, t_move as f64 / 1e3 / n_candidates as f64);
        println!("  CC+ComponentMasks: {:.1}ms ({:.1}µs/call)", t_comp as f64 / 1e6, t_comp as f64 / 1e3 / n_candidates as f64);
        println!("  get_best_move eval: {:.1}ms ({:.1}µs/call)", t_gbm as f64 / 1e6, t_gbm as f64 / 1e3 / n_candidates as f64);
    }

    // ── Full tournament move ──
    println!("\n--- Full tournament move (50 rollouts) ---");
    let mut g50 = game.clone();
    let t0 = Instant::now();
    let _ = tournament_player(&mut g50, 50, 20, 3.23);
    let t_tour = t0.elapsed();
    println!("  {:.1}ms/move ({:.1} mv/s)", t_tour.as_secs_f64() * 1e3, 1.0 / t_tour.as_secs_f64());

    println!("\n--- Full tournament move (200 rollouts) ---");
    let mut g200 = game.clone();
    let t0 = Instant::now();
    let _ = tournament_player(&mut g200, 200, 20, 3.23);
    let t_tour200 = t0.elapsed();
    println!("  {:.1}ms/move ({:.1} mv/s)", t_tour200.as_secs_f64() * 1e3, 1.0 / t_tour200.as_secs_f64());

    println!("\nTarget: 25 mv/s = 40ms/move");
}
