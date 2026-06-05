"""Color Lines 98 — Pygame GUI with AI Hint + Auto-play.

Run:
    python play_gui.py
    python play_gui.py --model alphatrain/data/sharp_25_epoch_12.pt

Controls:
    Click ball, then click reachable empty cell to move.
    Click selected ball again to deselect.
    [Undo]      revert the last move (one step only)
    [AI Hint]   highlight model's suggested source + target (policy argmax)
    [Auto Play] model plays continuously, ~1s per move (click again to stop)
    [New Game]  reset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pygame

# Allow running from repo root with no PYTHONPATH setup.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.board import ColorLinesGame  # noqa: E402
from game.config import BOARD_SIZE, NUM_COLORS  # noqa: E402

# ── Layout ────────────────────────────────────────────────────────────────
CELL = 56
BOARD_PX = CELL * BOARD_SIZE
MARGIN = 24
TOP_BAR_H = 64
BOTTOM_BAR_H = 80
RIGHT_PANEL_W = 200

WIN_W = MARGIN + BOARD_PX + MARGIN + RIGHT_PANEL_W + MARGIN
WIN_H = TOP_BAR_H + BOARD_PX + BOTTOM_BAR_H + 2 * MARGIN

BOARD_X = MARGIN
BOARD_Y = TOP_BAR_H + MARGIN
PANEL_X = BOARD_X + BOARD_PX + MARGIN
PANEL_Y = BOARD_Y

# ── Colors ────────────────────────────────────────────────────────────────
BG = (28, 28, 32)
GRID_DARK = (50, 50, 58)
GRID_LIGHT = (60, 60, 68)
TEXT = (220, 220, 220)
TEXT_DIM = (140, 140, 150)
SELECT_RING = (255, 230, 80)
REACHABLE_FILL = (60, 110, 70)
UNREACHABLE_FILL = (90, 50, 50)
HINT_SRC_RING = (80, 200, 255)
HINT_TGT_RING = (80, 255, 180)
LAST_MOVE_RING = (180, 180, 200)
BTN_BG = (60, 60, 72)
BTN_BG_HOVER = (80, 80, 96)
BTN_BG_ACTIVE = (90, 130, 180)
BTN_BG_DISABLED = (40, 40, 46)
BTN_FG = (235, 235, 240)
BTN_FG_DISABLED = (110, 110, 120)

BALL_COLORS = [
    None,
    (235, 60, 60),    # 1 red
    (70, 180, 70),    # 2 green
    (60, 110, 235),   # 3 blue
    (240, 215, 70),   # 4 yellow
    (80, 200, 220),   # 5 cyan
    (210, 80, 200),   # 6 magenta
    (160, 82, 45),    # 7 brown (distinguishable from yellow; orange was too close)
]
assert len(BALL_COLORS) >= NUM_COLORS + 1

AUTOPLAY_INTERVAL_S = 1.0
HINT_FADE_AFTER_S = 8.0


# ── Model loader ──────────────────────────────────────────────────────────
def _try_load_model(model_path):
    """Return (player_fn, device_str) or (None, msg) on failure."""
    try:
        import torch
        from alphatrain.evaluate import load_model, make_policy_player
    except ImportError as e:
        return None, f"AI disabled: {e}"

    if not os.path.exists(model_path):
        return None, f"AI disabled: model not found at {model_path}"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    try:
        net, _ = load_model(model_path, device, fp16=False)
        player = make_policy_player(net, device)
    except Exception as e:  # noqa: BLE001
        return None, f"AI disabled: {e}"

    return player, str(device)


# ── UI elements ───────────────────────────────────────────────────────────
class Button:
    def __init__(self, label, rect, on_click, hotkey=None):
        self.label = label
        self.rect = pygame.Rect(rect)
        self.on_click = on_click
        self.hotkey = hotkey
        self.enabled = True
        self.active = False  # toggled state (e.g. Auto Play)

    def draw(self, surf, font, mouse_pos):
        hover = self.rect.collidepoint(mouse_pos)
        if not self.enabled:
            bg = BTN_BG_DISABLED
            fg = BTN_FG_DISABLED
        elif self.active:
            bg = BTN_BG_ACTIVE
            fg = BTN_FG
        elif hover:
            bg = BTN_BG_HOVER
            fg = BTN_FG
        else:
            bg = BTN_BG
            fg = BTN_FG
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        txt = font.render(self.label, True, fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_click(self, pos):
        if self.enabled and self.rect.collidepoint(pos):
            self.on_click()
            return True
        return False


# ── Game wrapper with undo ────────────────────────────────────────────────
class GameState:
    def __init__(self, seed=None):
        self.game = ColorLinesGame(seed=seed)
        self.game.reset()  # spawn initial balls — __init__ alone leaves board empty
        self.snapshot = None  # for undo (one step)
        self.last_move = None  # (src, tgt) — for visualization
        self.last_result = None  # dict from game.move() — for status messages

    def _take_snapshot(self):
        g = self.game
        self.snapshot = {
            'board': g.board.copy(),
            'next_balls': [(tuple(rc), int(c)) for rc, c in g.next_balls],
            'score': g.score,
            'turns': g.turns,
            'game_over': g.game_over,
            'last_move': self.last_move,
        }

    def can_undo(self):
        return self.snapshot is not None

    def undo(self):
        if self.snapshot is None:
            return
        s = self.snapshot
        # Reset and restore — ColorLinesGame.reset() seeds new next_balls
        # if we don't pass them, so we pass them explicitly.
        self.game.reset(board=s['board'].copy(),
                         next_balls=list(s['next_balls']))
        self.game.score = s['score']
        self.game.turns = s['turns']
        self.game.game_over = s['game_over']
        self.last_move = s['last_move']
        self.snapshot = None  # only one level of undo

    def move(self, src, tgt):
        if self.game.game_over:
            return False
        self._take_snapshot()
        result = self.game.move(src, tgt)
        if not result['valid']:
            self.snapshot = None  # rollback unused snapshot
            return False
        self.last_move = (src, tgt)
        self.last_result = result
        return True


class _FakeGame:
    """Minimal duck-typed stand-in so the draw_* helpers work in replay."""
    board = None
    next_balls = []
    score = 0
    turns = 0
    game_over = False


class ReplayState:
    """Steps through a recorded game (scripts/find_worst_game.py output).

    Each frame stores the board+next_balls BEFORE the move, plus the chosen
    move and per-turn debug (empties, LEC, n_components, top-K policy).
    Frames are pre-computed; no engine/RNG is run during replay.
    """

    def __init__(self, path):
        with open(path) as fh:
            self.data = json.load(fh)
        self.frames = self.data['frames']
        if not self.frames:
            raise ValueError(f"No frames in {path}")
        self.idx = 0
        self.game = _FakeGame()
        self.last_move = None
        self.last_result = None
        self.cur = None
        self._load()

    def _load(self):
        fr = self.frames[self.idx]
        self.cur = fr
        self.game = _FakeGame()
        self.game.board = np.array(fr['board'], dtype=np.int8)
        self.game.next_balls = [(tuple(rc), int(c))
                                 for rc, c in fr['next_balls']]
        self.game.score = fr.get('score_before', fr['score'])
        self.game.turns = fr['turn']
        # game_over flag only meaningful on the last frame
        self.game.game_over = (self.idx == len(self.frames) - 1
                                and self.data.get('died', False))
        cm = fr.get('chosen_move')
        self.last_move = ((tuple(cm[0]), tuple(cm[1])) if cm else None)

    def next(self):
        if self.idx < len(self.frames) - 1:
            self.idx += 1
            self._load()

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load()

    def first(self):
        self.idx = 0
        self._load()

    def last(self):
        self.idx = len(self.frames) - 1
        self._load()

    def jump(self, frac):
        self.idx = max(0, min(len(self.frames) - 1,
                              int(frac * (len(self.frames) - 1))))
        self._load()


# ── Drawing ───────────────────────────────────────────────────────────────
def cell_rect(r, c):
    return pygame.Rect(BOARD_X + c * CELL, BOARD_Y + r * CELL, CELL, CELL)


def cell_center(r, c):
    return (BOARD_X + c * CELL + CELL // 2,
            BOARD_Y + r * CELL + CELL // 2)


def screen_to_cell(x, y):
    if not (BOARD_X <= x < BOARD_X + BOARD_PX and
            BOARD_Y <= y < BOARD_Y + BOARD_PX):
        return None
    return ((y - BOARD_Y) // CELL, (x - BOARD_X) // CELL)


def draw_board(surf, gs, selected, target_mask, hint, font_small):
    # Board background
    pygame.draw.rect(surf, GRID_DARK,
                     (BOARD_X - 2, BOARD_Y - 2,
                      BOARD_PX + 4, BOARD_PX + 4),
                     border_radius=4)

    # Cells
    board = gs.game.board
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rect = cell_rect(r, c)
            base_color = GRID_LIGHT if (r + c) % 2 == 0 else GRID_DARK
            # Tint empty cells based on reachability when source selected
            if (selected is not None and board[r, c] == 0
                    and target_mask is not None):
                base_color = (REACHABLE_FILL if target_mask[r, c] > 0
                              else UNREACHABLE_FILL)
            pygame.draw.rect(surf, base_color, rect)

    # Last-move highlight (subtle)
    if gs.last_move is not None:
        for r, c in (gs.last_move[0], gs.last_move[1]):
            pygame.draw.rect(surf, LAST_MOVE_RING,
                             cell_rect(r, c), width=2, border_radius=3)

    # Balls
    radius = CELL // 2 - 6
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            col = int(board[r, c])
            if col == 0:
                continue
            pygame.draw.circle(surf, BALL_COLORS[col], cell_center(r, c),
                                radius)
            # Subtle highlight on ball
            pygame.draw.circle(surf, (255, 255, 255, 60),
                                (cell_center(r, c)[0] - radius // 3,
                                 cell_center(r, c)[1] - radius // 3),
                                radius // 4)

    # Next-ball previews (diamonds on their spawn squares, faded interior)
    for (r, c), col in gs.game.next_balls:
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == 0:
            cx, cy = cell_center(r, c)
            d = 9
            pts = [(cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)]
            faded = tuple(v // 3 for v in BALL_COLORS[col])
            pygame.draw.polygon(surf, faded, pts)
            pygame.draw.polygon(surf, BALL_COLORS[col], pts, width=2)

    # Selection ring
    if selected is not None:
        pygame.draw.circle(surf, SELECT_RING, cell_center(*selected),
                            radius + 4, width=3)

    # AI Hint
    if hint is not None:
        src, tgt = hint
        pygame.draw.circle(surf, HINT_SRC_RING, cell_center(*src),
                            radius + 6, width=3)
        pygame.draw.rect(surf, HINT_TGT_RING, cell_rect(*tgt),
                          width=3, border_radius=3)

    # Grid lines (light)
    for i in range(BOARD_SIZE + 1):
        pygame.draw.line(surf, (40, 40, 46),
                         (BOARD_X, BOARD_Y + i * CELL),
                         (BOARD_X + BOARD_PX, BOARD_Y + i * CELL))
        pygame.draw.line(surf, (40, 40, 46),
                         (BOARD_X + i * CELL, BOARD_Y),
                         (BOARD_X + i * CELL, BOARD_Y + BOARD_PX))


def draw_top_bar(surf, gs, ai_status, font_big, font_small):
    y = MARGIN // 2
    txt = font_big.render(f"Score: {gs.game.score}", True, TEXT)
    surf.blit(txt, (MARGIN, y))
    turn_txt = font_small.render(f"Turn: {gs.game.turns}", True, TEXT_DIM)
    surf.blit(turn_txt, (MARGIN + 220, y + 8))
    if gs.game.game_over:
        over = font_big.render("GAME OVER", True, (240, 80, 80))
        surf.blit(over, (MARGIN + 380, y))
    # AI status (right-aligned)
    ai_txt = font_small.render(ai_status, True, TEXT_DIM)
    surf.blit(ai_txt, (WIN_W - MARGIN - ai_txt.get_width(), y + 8))


def draw_status_message(surf, gs, autoplay, font_small):
    """Single line below the board: last-move feedback / autoplay state."""
    if gs.game.game_over:
        msg, color = "Game over — click New Game to restart", (240, 80, 80)
    elif gs.last_result is not None and gs.last_result.get('cleared', 0) > 0:
        c = gs.last_result['cleared']
        s = gs.last_result['score']
        msg, color = f"✦ Cleared {c} balls (+{s} pts)", (240, 220, 80)
    elif autoplay:
        msg, color = "Auto-playing — click Stop to interrupt", (140, 200, 255)
    elif gs.last_move is not None:
        (sr, sc), (tr, tc) = gs.last_move
        msg = f"Moved ({sr},{sc}) → ({tr},{tc})"
        color = TEXT_DIM
    else:
        msg, color = "Click a ball, then click a reachable empty cell to move", TEXT_DIM
    txt = font_small.render(msg, True, color)
    surf.blit(txt, (BOARD_X, BOARD_Y + BOARD_PX + 8))


def draw_right_panel(surf, gs, font_small):
    title = font_small.render("Next balls", True, TEXT_DIM)
    surf.blit(title, (PANEL_X, PANEL_Y))
    radius = 18
    for i, ((r, c), col) in enumerate(gs.game.next_balls):
        cx = PANEL_X + radius + 4
        cy = PANEL_Y + 30 + i * (2 * radius + 14)
        pygame.draw.circle(surf, BALL_COLORS[col], (cx, cy), radius)
        loc_txt = font_small.render(f"@ ({r}, {c})", True, TEXT_DIM)
        surf.blit(loc_txt, (cx + radius + 12, cy - 8))


def _empty_component_stats(board):
    """Largest empty connected component size + number of empty components.

    Computed from the board so the panel works for leaner recordings (e.g.
    batch_record) that don't store 'lec'/'n_components' per frame.
    """
    seen = [[False] * 9 for _ in range(9)]
    largest = n = 0
    for r0 in range(9):
        for c0 in range(9):
            if board[r0][c0] != 0 or seen[r0][c0]:
                continue
            n += 1
            size = 0
            stack = [(r0, c0)]
            seen[r0][c0] = True
            while stack:
                r, c = stack.pop()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 9 and 0 <= nc < 9
                            and not seen[nr][nc] and board[nr][nc] == 0):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            largest = max(largest, size)
    return largest, n


def draw_replay_panel(surf, rs, font_small, font):
    """Debug panel for replay mode: frame nav + per-turn metrics + top-K."""
    fr = rs.cur
    x = PANEL_X
    y = PANEL_Y
    n = len(rs.frames)

    def line(text, color=TEXT, dy=18, fnt=font_small):
        nonlocal y
        surf.blit(fnt.render(text, True, color), (x, y))
        y += dy

    line(f"REPLAY  seed {rs.data.get('seed', '?')}", (140, 200, 255), 20)
    model = os.path.basename(rs.data.get('model', ''))
    line(model, TEXT_DIM, 22)

    line(f"Frame {rs.idx + 1}/{n}", TEXT, 18)
    line(f"Turn {fr['turn']}", TEXT, 18)
    delta = fr['score'] - fr.get('score_before', fr['score'])
    sc_txt = f"Score {fr.get('score_before', fr['score'])}"
    if delta:
        sc_txt += f"  (+{delta})"
    line(sc_txt, (240, 220, 80) if delta else TEXT, 22)

    # Fragmentation metrics — color LEC by danger. Compute from the board when
    # the recording didn't store them (leaner batch_record frames).
    if 'lec' in fr and 'n_components' in fr:
        lec, n_comp = fr['lec'], fr['n_components']
    else:
        lec, n_comp = _empty_component_stats(fr['board'])
    empties = fr.get('empties', sum(v == 0 for row in fr['board'] for v in row))
    lec_color = ((240, 80, 80) if lec < 10
                 else (240, 200, 80) if lec < 20 else (120, 220, 120))
    line(f"Empties: {empties}", TEXT_DIM, 18)
    line(f"LEC: {lec}", lec_color, 18)
    line(f"Components: {n_comp}", TEXT_DIM, 22)

    # Chosen move
    cm = fr.get('chosen_move')
    if cm:
        line(f"Move ({cm[0][0]},{cm[0][1]})->({cm[1][0]},{cm[1][1]})",
             (80, 255, 180), 18)
    res = fr.get('result')
    if res and res.get('cleared'):
        line(f"  cleared {res['cleared']} (+{res['score']})",
             (240, 220, 80), 18)
    if fr.get('no_legal_move'):
        line("NO LEGAL MOVE — death", (240, 80, 80), 18)
    y += 6

    # Top-K policy
    line("Top-K policy:", TEXT_DIM, 18)
    chosen_t = tuple(map(tuple, cm)) if cm else None
    for c in fr.get('top_k', [])[:8]:
        mv = c['move']
        mvt = (tuple(mv[0]), tuple(mv[1]))
        is_chosen = (mvt == chosen_t)
        col = (80, 255, 180) if is_chosen else TEXT_DIM
        bar = '#' * int(round(c['prob'] * 20))
        line(f"{c['prob']:.2f} {bar}", col, 15)


def draw_replay_help(surf, font_small):
    msg = ("[<- / ->] step   [Home/End] ends   "
           "[Space] +10   [B] -10   [Esc] quit")
    txt = font_small.render(msg, True, TEXT_DIM)
    surf.blit(txt, (BOARD_X, BOARD_Y + BOARD_PX + 8))


def replay_main(screen, clock, fonts, replay_path, start_frame=0):
    """Replay loop: step through a recorded game with debug overlay."""
    font_big, font, font_small = fonts
    rs = ReplayState(replay_path)
    if start_frame:
        rs.idx = max(0, min(int(start_frame), len(rs.frames) - 1))
        rs._load()
    print(f"Replay: {replay_path} — {len(rs.frames)} frames, "
          f"seed={rs.data.get('seed')}, final_score={rs.data.get('final_score')}, "
          f"died={rs.data.get('died')}", flush=True)

    btn_w, btn_h = 90, 40
    by = BOARD_Y + BOARD_PX + (BOTTOM_BAR_H - btn_h) // 2
    gap = 10
    bx = MARGIN
    btn_first = Button("|<", (bx, by, 50, btn_h), rs.first)
    bx += 50 + gap
    btn_prev = Button("< Prev", (bx, by, btn_w, btn_h), rs.prev)
    bx += btn_w + gap
    btn_next = Button("Next >", (bx, by, btn_w, btn_h), rs.next)
    bx += btn_w + gap
    btn_last = Button(">|", (bx, by, 50, btn_h), rs.last)
    buttons = [btn_first, btn_prev, btn_next, btn_last]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_RIGHT, pygame.K_n):
                    rs.next()
                elif event.key in (pygame.K_LEFT, pygame.K_p):
                    rs.prev()
                elif event.key == pygame.K_HOME:
                    rs.first()
                elif event.key == pygame.K_END:
                    rs.last()
                elif event.key == pygame.K_SPACE:
                    for _ in range(10):
                        rs.next()
                elif event.key == pygame.K_b:
                    for _ in range(10):
                        rs.prev()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for b in buttons:
                    if b.handle_click(event.pos):
                        break

        screen.fill(BG)
        draw_top_bar(screen, rs, "REPLAY MODE", font_big, font_small)
        # Highlight the chosen move via the hint rings (src=blue, tgt=green)
        draw_board(screen, rs, None, None, rs.last_move, font_small)
        draw_replay_help(screen, font_small)
        draw_replay_panel(screen, rs, font_small, font)
        mouse_pos = pygame.mouse.get_pos()
        for b in buttons:
            b.draw(screen, font, mouse_pos)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                         default='alphatrain/data/sharp_25_epoch_12.pt',
                         help='Path to PolicyNet checkpoint for AI features.')
    parser.add_argument('--seed', type=int, default=None,
                         help='Initial RNG seed (omit for random).')
    parser.add_argument('--replay', type=str, default=None,
                         help='Path to a recorded game JSON (from '
                              'scripts/find_worst_game.py). Enters replay '
                              'mode with debug overlay + next/prev.')
    parser.add_argument('--replay-frame', type=int, default=0,
                         help='Jump directly to this frame on open '
                              '(e.g. a fork frame to inspect).')
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("Color Lines 98")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont('Helvetica', 24, bold=True)
    font = pygame.font.SysFont('Helvetica', 16)
    font_small = pygame.font.SysFont('Helvetica', 13)

    if args.replay:
        pygame.display.set_caption("Color Lines 98 — Replay")
        replay_main(screen, clock, (font_big, font, font_small), args.replay,
                    start_frame=args.replay_frame)
        return

    ai_player, ai_status = _try_load_model(args.model)
    if ai_player is None:
        print(ai_status)
    else:
        ai_status = f"AI: {os.path.basename(args.model)} [{ai_status}]"
        print(ai_status)

    gs = GameState(seed=args.seed)
    selected = None  # (r, c) or None
    target_mask = None  # ndarray when selected, else None
    hint = None
    hint_time = 0.0
    autoplay = False
    last_autoplay_t = 0.0

    def new_game():
        nonlocal gs, selected, target_mask, hint, autoplay
        gs = GameState(seed=None)
        selected = None
        target_mask = None
        hint = None
        autoplay = False
        btn_autoplay.active = False
        btn_autoplay.label = "Auto Play"

    def do_undo():
        nonlocal selected, target_mask, hint, autoplay
        if gs.can_undo():
            gs.undo()
            selected = None
            target_mask = None
            hint = None
            # Stop autoplay if it was running — user undid an AI move
            autoplay = False
            btn_autoplay.active = False
            btn_autoplay.label = "Auto Play"

    def do_hint():
        nonlocal hint, hint_time
        if ai_player is None or gs.game.game_over:
            return
        move = ai_player(gs.game)
        if move is not None:
            hint = move
            hint_time = time.time()

    def toggle_autoplay():
        nonlocal autoplay, last_autoplay_t
        if ai_player is None or gs.game.game_over:
            return
        autoplay = not autoplay
        btn_autoplay.active = autoplay
        btn_autoplay.label = "Stop" if autoplay else "Auto Play"
        if autoplay:
            last_autoplay_t = 0.0  # play immediately on first tick

    # Buttons — bottom bar
    btn_w, btn_h = 130, 40
    by = BOARD_Y + BOARD_PX + (BOTTOM_BAR_H - btn_h) // 2
    bx0 = MARGIN
    gap = 12
    btn_new = Button("New Game", (bx0, by, btn_w, btn_h), new_game,
                      hotkey=pygame.K_n)
    btn_undo = Button("Undo", (bx0 + (btn_w + gap), by, btn_w, btn_h),
                       do_undo, hotkey=pygame.K_u)
    btn_hint = Button("AI Hint", (bx0 + 2 * (btn_w + gap), by, btn_w, btn_h),
                       do_hint, hotkey=pygame.K_h)
    btn_autoplay = Button("Auto Play",
                           (bx0 + 3 * (btn_w + gap), by, btn_w, btn_h),
                           toggle_autoplay, hotkey=pygame.K_a)
    buttons = [btn_new, btn_undo, btn_hint, btn_autoplay]

    running = True
    while running:
        # ── enable/disable AI-dependent buttons each frame ──
        btn_undo.enabled = gs.can_undo()
        btn_hint.enabled = ai_player is not None and not gs.game.game_over
        btn_autoplay.enabled = ai_player is not None and not gs.game.game_over
        if gs.game.game_over and autoplay:
            autoplay = False
            btn_autoplay.active = False
            btn_autoplay.label = "Auto Play"

        # ── autoplay tick ──
        now = time.time()
        if autoplay and (now - last_autoplay_t) >= AUTOPLAY_INTERVAL_S:
            move = ai_player(gs.game)
            if move is None:
                autoplay = False
                btn_autoplay.active = False
                btn_autoplay.label = "Auto Play"
            else:
                gs.move(*move)
                selected = None
                target_mask = None
                hint = None
                last_autoplay_t = now

        # ── hint fade ──
        if hint is not None and (now - hint_time) > HINT_FADE_AFTER_S:
            hint = None

        # ── events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    for b in buttons:
                        if b.hotkey == event.key and b.enabled:
                            b.on_click()
                            break
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handled = False
                for b in buttons:
                    if b.handle_click(event.pos):
                        handled = True
                        break
                if handled:
                    continue
                if autoplay:  # ignore board clicks while autoplaying
                    continue
                cell = screen_to_cell(*event.pos)
                if cell is None or gs.game.game_over:
                    continue
                r, c = cell
                board = gs.game.board
                if selected is None:
                    # Try to select a ball
                    if board[r, c] != 0:
                        src_mask = gs.game.get_source_mask()
                        if src_mask[r, c] > 0:
                            selected = (r, c)
                            target_mask = gs.game.get_target_mask(selected)
                else:
                    if (r, c) == selected:
                        selected = None
                        target_mask = None
                    elif board[r, c] != 0:
                        # Re-select a different ball
                        src_mask = gs.game.get_source_mask()
                        if src_mask[r, c] > 0:
                            selected = (r, c)
                            target_mask = gs.game.get_target_mask(selected)
                    else:
                        # Try move to empty target
                        if (target_mask is not None
                                and target_mask[r, c] > 0):
                            if gs.move(selected, (r, c)):
                                hint = None  # invalidate any pending hint
                            selected = None
                            target_mask = None

        # ── draw ──
        screen.fill(BG)
        draw_top_bar(screen, gs, ai_status, font_big, font_small)
        draw_board(screen, gs, selected, target_mask, hint, font_small)
        draw_status_message(screen, gs, autoplay, font_small)
        draw_right_panel(screen, gs, font_small)
        mouse_pos = pygame.mouse.get_pos()
        for b in buttons:
            b.draw(screen, font, mouse_pos)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
