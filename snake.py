import pygame
import random
import torch
import collections
import math
import sys

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
CELL        = 28
COLS        = 24
ROWS        = 24
GRID_W      = COLS * CELL          # 672
GRID_H      = ROWS * CELL          # 672
PANEL_W     = 400
WIN_H       = 900                  # taller than grid — extra space for charts
WIN_W       = GRID_W + PANEL_W
FPS         = 60
TICK_MS     = 15                   # game step every 15 ms  (≈ turtle ontimer)

# Colours
C_BG        = (10,  12,  18)
C_GRID      = (38,  44,  62)
C_DIVIDER   = (60,  66,  90)
C_HEAD      = (200, 255, 200)
C_BODY      = (80,  200, 100)
C_FOOD      = (255,  70,  70)
C_FOOD_GLOW = (255, 120, 120)
C_PANEL_BG  = (14,  16,  24)
C_WHITE     = (220, 224, 235)
C_DIM       = (130, 135, 160)
C_GREEN     = ( 0,  255,  65)
C_YELLOW    = (255, 210,   0)
C_ORANGE    = (255, 153,  68)
C_RED       = (255,  68,  68)
C_BLUE      = (130, 190, 255)
C_SECTION   = (110, 116, 145)

# ─────────────────────────────────────────────
#  AI / DQN backend  (untouched logic)
# ─────────────────────────────────────────────
atminties_buferis = collections.deque(maxlen=100_000)

class SnakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sluoksnis1 = torch.nn.Linear(11, 128)
        self.sluoksnis2 = torch.nn.Linear(128, 64)
        self.sluoksnis3 = torch.nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.sluoksnis1(x))
        x = torch.relu(self.sluoksnis2(x))
        return self.sluoksnis3(x)

tinklas       = SnakeNet()
tikslo_tinklas = SnakeNet()
tikslo_tinklas.load_state_dict(tinklas.state_dict())
optimizer     = torch.optim.Adam(tinklas.parameters(), lr=0.0001)

TARGET_UPDATE_STEPS = 1000

def _net_param_count(model):
    return sum(p.numel() for p in model.parameters())

# ─────────────────────────────────────────────
#  Game state
# ─────────────────────────────────────────────
score       = 0
moves       = 0
atlygis     = 0
deaths      = 0
episode     = 0
last_loss   = 0.0
best_score  = 0
epsilon     = 0.9
total_steps = 0
game_running = True

# History for charts (capped so old data scrolls off)
HISTORY_LEN   = 200
score_history = collections.deque(maxlen=HISTORY_LEN)   # score per episode
loss_history  = collections.deque(maxlen=HISTORY_LEN)   # loss sample per step

# Snake is a list of [col, row] positions; index 0 = head
snake_body  = [[11, 11]]          # start near centre
heading     = 0                   # 0=R, 90=U, 180=L, 270=D  (matches turtle)
food_pos    = [0, 0]              # [col, row]

def _heading_to_delta(h):
    return {0: (1, 0), 90: (0, -1), 180: (-1, 0), 270: (0, 1)}[h]

def place_food():
    global food_pos
    occupied = set(map(tuple, snake_body))
    while True:
        c = random.randint(0, COLS - 1)
        r = random.randint(0, ROWS - 1)
        if (c, r) not in occupied:
            food_pos = [c, r]
            return

# ─────────────────────────────────────────────
#  State / AI helpers  (logic unchanged)
# ─────────────────────────────────────────────
def get_state():
    hc, hr = snake_body[0]
    fc, fr = food_pos
    h      = heading
    body_set = set(map(tuple, snake_body[1:]))

    def blocked(dc, dr):
        nc, nr = hc + dc, hr + dr
        return nc < 0 or nc >= COLS or nr < 0 or nr >= ROWS or (nc, nr) in body_set

    if h == 0:
        front = blocked(1, 0); left = blocked(0, -1); right = blocked(0, 1)
    elif h == 90:
        front = blocked(0, -1); left = blocked(-1, 0); right = blocked(1, 0)
    elif h == 180:
        front = blocked(-1, 0); left = blocked(0, 1); right = blocked(0, -1)
    else:  # 270
        front = blocked(0, 1); left = blocked(1, 0); right = blocked(-1, 0)

    return [
        int(fc > hc), int(fc < hc), int(fr < hr), int(fr > hr),
        int(front), int(left), int(right),
        int(h == 0), int(h == 90), int(h == 180), int(h == 270),
    ]

def judesio_pasirinkimas():
    global heading
    busena     = get_state()
    q_reiksmes = tinklas(torch.FloatTensor([busena]))

    if random.random() < epsilon:
        veiksmas = random.randint(0, 3)
    else:
        veiksmas = q_reiksmes.argmax().item()

    nauja_kryptis = veiksmas * 90
    if abs(heading - nauja_kryptis) != 180:
        heading = nauja_kryptis

def atmintis(buvusi_busena):
    global atlygis
    veiksmas           = heading // 90
    dabartinis_atlygis = atlygis
    nauja_busena       = get_state()
    atminties_buferis.append((buvusi_busena, veiksmas, dabartinis_atlygis, nauja_busena))
    atlygis = 0
    
def learning():
    global last_loss, total_steps
    if len(atminties_buferis) < 1500:
        return
    partija = random.sample(atminties_buferis, 32)
    busenos, veiksmai, atl, naujos_busenos = zip(*partija)

    busenos        = torch.FloatTensor(busenos)
    veiksmai       = torch.LongTensor(veiksmai)
    atl            = torch.FloatTensor(atl)
    naujos_busenos = torch.FloatTensor(naujos_busenos)

    rezultatas   = tinklas(busenos)
    q_dabartinis = rezultatas.gather(1, veiksmai.unsqueeze(1))
    q_kitas      = tikslo_tinklas(naujos_busenos)
    q_tikslas    = (q_kitas.max(1)[0] * 0.99 + atl).detach()
    klaida       = torch.nn.functional.mse_loss(q_dabartinis, q_tikslas.unsqueeze(1))

    optimizer.zero_grad()
    klaida.backward()
    torch.nn.utils.clip_grad_norm_(tinklas.parameters(), 1.0)
    optimizer.step()
    last_loss = klaida.item()
    loss_history.append(last_loss)

    total_steps += 1
    if total_steps % TARGET_UPDATE_STEPS == 0:
        tikslo_tinklas.load_state_dict(tinklas.state_dict())

def reset():
    global score, moves, atlygis, game_running, deaths, episode, best_score, epsilon
    global snake_body, heading
    if score > best_score:
        best_score = score
    score_history.append(score)
    deaths  += 1
    episode += 1
    epsilon  = max(0.05, epsilon * 0.995)
    score    = 0
    moves    = 0
    atlygis  = 0
    game_running = True
    snake_body   = [[11, 11]]
    heading      = 0
    place_food()

def game_step():
    global score, moves, atlygis, game_running

    moves += 1
    judesio_pasirinkimas()

    busena_pries = get_state()

    if moves > 250 * len(snake_body):
        atlygis -= 10
        atmintis(busena_pries)
        learning()
        reset()
        return

    hc, hr = snake_body[0]
    dc, dr = _heading_to_delta(heading)
    nc, nr = hc + dc, hr + dr

    body_set = set(map(tuple, snake_body[1:]))

    if nc < 0 or nc >= COLS or nr < 0 or nr >= ROWS or (nc, nr) in body_set:
        atlygis -= 10
        atmintis(busena_pries)
        learning()
        reset()
        return

    snake_body.insert(0, [nc, nr])

    if [nc, nr] == food_pos:
        score   += 1
        atlygis += 10
        place_food()
    else:
        snake_body.pop()

    atmintis(busena_pries)
    learning()

# ─────────────────────────────────────────────
#  Pygame rendering
# ─────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Snake AI — DQN")
clock  = pygame.time.Clock()

FONT_LABEL    = pygame.font.SysFont("Courier New", 12)
FONT_VALUE    = pygame.font.SysFont("Courier New", 22, bold=True)
FONT_VALUE_SM = pygame.font.SysFont("Courier New", 17, bold=True)
FONT_SECTION  = pygame.font.SysFont("Courier New", 11, bold=True)
FONT_TITLE    = pygame.font.SysFont("Courier New", 24, bold=True)

def draw_grid(surf):
    for c in range(COLS + 1):
        x = c * CELL
        pygame.draw.line(surf, C_GRID, (x, 0), (x, GRID_H), 2)
    for r in range(ROWS + 1):
        y = r * CELL
        pygame.draw.line(surf, C_GRID, (0, y), (GRID_W, y), 2)

def draw_snake(surf):
    for i, (c, r) in enumerate(snake_body):
        x, y = c * CELL, r * CELL
        color = C_HEAD if i == 0 else C_BODY
        rect  = pygame.Rect(x + 2, y + 2, CELL - 4, CELL - 4)
        pygame.draw.rect(surf, color, rect, border_radius=4)
        if i == 0:
            # subtle highlight dot on head
            pygame.draw.rect(surf, (255, 255, 255, 80), pygame.Rect(x + 5, y + 5, 4, 4), border_radius=2)

def draw_food(surf, tick):
    fc, fr = food_pos
    cx, cy = fc * CELL + CELL // 2, fr * CELL + CELL // 2
    pulse  = int(4 + 3 * math.sin(tick * 0.08))
    pygame.draw.circle(surf, C_FOOD_GLOW, (cx, cy), pulse + 4)
    pygame.draw.circle(surf, C_FOOD,      (cx, cy), pulse)

def draw_divider(surf):
    pygame.draw.line(surf, C_DIVIDER, (GRID_W, 0), (GRID_W, WIN_H), 2)

# ─── Stats panel ───────────────────────────────
PAD   = 16   # horizontal padding inside panel
INNER = GRID_W + PAD
INNER_R = GRID_W + PANEL_W - PAD

def _t(surf, txt, x, y, font, color, align="left"):
    s = font.render(str(txt), True, color)
    if align == "right":   x -= s.get_width()
    elif align == "center": x -= s.get_width() // 2
    surf.blit(s, (x, y))

def _section_label(surf, txt, y):
    """Uppercase dim section divider with a line."""
    _t(surf, txt, INNER, y, FONT_SECTION, C_SECTION)
    lx = INNER + FONT_SECTION.size(txt)[0] + 8
    pygame.draw.line(surf, C_DIVIDER, (lx, y + 6), (INNER_R, y + 6), 1)

def _card(surf, label, value, x, y, w, val_color=C_WHITE):
    """Card: big value on top, small label below, inside a dark rounded rect."""
    card_h = 52
    card_rect = pygame.Rect(x, y, w, card_h)
    pygame.draw.rect(surf, (22, 25, 36), card_rect, border_radius=6)
    cx = x + w // 2
    _t(surf, str(value), cx, y + 5,  FONT_VALUE,   val_color, align="center")
    _t(surf, label,      cx, y + 33, FONT_LABEL,   C_DIM,     align="center")

def _card_sm(surf, label, value, x, y, w, val_color=C_WHITE):
    """Smaller card for denser sections."""
    card_h = 42
    card_rect = pygame.Rect(x, y, w, card_h)
    pygame.draw.rect(surf, (22, 25, 36), card_rect, border_radius=6)
    cx = x + w // 2
    _t(surf, str(value), cx, y + 4,  FONT_VALUE_SM, val_color, align="center")
    _t(surf, label,      cx, y + 27, FONT_LABEL,    C_DIM,     align="center")

def _bar_card(surf, label, ratio, display_val, x, y, w, bar_color):
    """Card with a progress bar inside instead of a plain number."""
    card_h = 52
    pygame.draw.rect(surf, (22, 25, 36), pygame.Rect(x, y, w, card_h), border_radius=6)
    cx = x + w // 2
    _t(surf, str(display_val), cx, y + 5, FONT_VALUE, bar_color, align="center")
    bar_x = x + 8
    bar_w = w - 16
    bar_h = 6
    bar_y = y + 34
    pygame.draw.rect(surf, C_DIVIDER, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
    fill = int(bar_w * max(0.0, min(ratio, 1.0)))
    if fill > 0:
        pygame.draw.rect(surf, bar_color, (bar_x, bar_y, fill, bar_h), border_radius=3)
    _t(surf, label, cx, bar_y + 9, FONT_LABEL, C_DIM, align="center")

def _info_row(surf, label, value, y, val_color=C_DIM):
    """Compact single-line row for static network info."""
    _t(surf, label, INNER,   y, FONT_LABEL, C_DIM)
    _t(surf, str(value), INNER_R, y, FONT_LABEL, val_color, align="right")

def _draw_chart(surf, data, x, y, w, h, color, label, show_avg=True):
    """Minimal line chart. data = deque of floats."""
    # background
    pygame.draw.rect(surf, (18, 21, 32), (x, y, w, h), border_radius=4)
    pygame.draw.rect(surf, C_DIVIDER,    (x, y, w, h), 1, border_radius=4)

    if len(data) < 2:
        _t(surf, label,           x + 6,      y + 4,      FONT_LABEL, C_DIM)
        _t(surf, "collecting...", x + w // 2, y + h // 2, FONT_LABEL, C_SECTION, align="center")
        return

    vals   = list(data)
    lo, hi = min(vals), max(vals)
    span   = hi - lo if hi != lo else 1.0

    def _py(v):
        return y + h - 4 - int(((v - lo) / span) * (h - 12))

    # plot points as a polyline
    pts = [(x + int(i / max(len(vals) - 1, 1) * (w - 2)), _py(v))
           for i, v in enumerate(vals)]
    if len(pts) > 1:
        pygame.draw.lines(surf, color, False, pts, 1)

    # moving average (last 20)
    if show_avg and len(vals) >= 10:
        win = 20
        avgs = []
        for i in range(len(vals)):
            sl = vals[max(0, i - win + 1): i + 1]
            avgs.append(sum(sl) / len(sl))
        avg_pts = [(x + int(i / max(len(avgs) - 1, 1) * (w - 2)),
                    y + h - 4 - int(((v - lo) / span) * (h - 12)))
                   for i, v in enumerate(avgs)]
        if len(avg_pts) > 1:
            pygame.draw.lines(surf, C_YELLOW, False, avg_pts, 2)

    # axis labels
    _t(surf, label,        x + 5,     y + 3,      FONT_LABEL, C_DIM)
    _t(surf, f"{hi:.1f}",  x + w - 4, y + 3,      FONT_LABEL, C_WHITE, align="right")
    _t(surf, f"{lo:.1f}",  x + w - 4, y + h - 13, FONT_LABEL, C_WHITE, align="right")


def draw_panel(surf):
    pygame.draw.rect(surf, C_PANEL_BG, pygame.Rect(GRID_W, 0, PANEL_W, WIN_H))
    draw_divider(surf)

    cy = 14
    mid = GRID_W + PANEL_W // 2

    # ── Title ──────────────────────────────────
    _t(surf, "SNAKE AI", mid, cy, FONT_TITLE, C_GREEN, align="center")
    cy += 30
    pygame.draw.line(surf, C_DIVIDER, (INNER, cy), (INNER_R, cy), 1)
    cy += 12

    # ── GAME  (2-column card grid) ─────────────
    _section_label(surf, "GAME", cy); cy += 14

    gap   = 6
    col_w = (PANEL_W - PAD * 2 - gap) // 2
    col_a = INNER
    col_b = INNER + col_w + gap

    _card(surf, "SCORE",      score,           col_a, cy, col_w, C_GREEN)
    _card(surf, "BEST",       best_score,      col_b, cy, col_w, C_GREEN)
    cy += 58

    _card(surf, "EPISODE",    episode,         col_a, cy, col_w, C_WHITE)
    _card(surf, "DEATHS",     deaths,          col_b, cy, col_w, C_RED)
    cy += 58

    _card(surf, "MOVES",      moves,           col_a, cy, col_w, C_WHITE)
    _card(surf, "BODY LEN",   len(snake_body), col_b, cy, col_w, C_WHITE)
    cy += 62

    # ── RL / TRAINING ──────────────────────────
    _section_label(surf, "RL / TRAINING", cy); cy += 14

    col3_w = (PANEL_W - PAD * 2 - gap * 2) // 3
    col3_b = INNER + col3_w + gap
    col3_c = col3_b + col3_w + gap

    _bar_card(surf, "EPSILON", epsilon,
              f"{epsilon:.2f}", col_a, cy, col3_w, C_GREEN)
    _bar_card(surf, "LOSS",
              min(last_loss / 2.0, 1.0),
              f"{last_loss:.3f}", col3_b, cy, col3_w,
              C_GREEN if last_loss < 0.5 else C_ORANGE if last_loss < 1.0 else C_RED)
    _card_sm(surf, "STEPS", f"{total_steps:,}", col3_c, cy, col3_w, C_WHITE)
    cy += 62

    _card_sm(surf, "BUFFER",  f"{len(atminties_buferis):,}", col_a, cy, col_w, C_WHITE)
    _card_sm(surf, "BATCH/γ", f"32 / 0.99",                  col_b, cy, col_w, C_DIM)
    cy += 52

    # ── NETWORK ────────────────────────────────
    _section_label(surf, "NETWORK", cy); cy += 14
    info = [
        ("Architecture", "11 → 128 → 64 → 4", C_BLUE),
        ("Activation",   "ReLU",               C_BLUE),
        ("Optimizer",    "Adam  lr=0.001",      C_BLUE),
        ("Parameters",   f"{_net_param_count(tinklas):,}", C_WHITE),
        ("Inputs",       "11 features",         C_DIM),
        ("Outputs",      "4  (U / R / D / L)",  C_DIM),
    ]
    for lbl, val, col in info:
        _info_row(surf, lbl, val, cy, col); cy += 17
    cy += 10

    # ── Charts ─────────────────────────────────
    chart_w = PANEL_W - PAD * 2
    chart_h = 72
    _section_label(surf, "SCORE / EPISODE", cy); cy += 14
    _draw_chart(surf, score_history, INNER, cy, chart_w, chart_h, C_GREEN,  "score")
    cy += chart_h + 10

    _section_label(surf, "LOSS / STEP", cy); cy += 14
    _draw_chart(surf, loss_history,  INNER, cy, chart_w, chart_h, C_ORANGE, "loss", show_avg=False)

# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
place_food()

last_step_time = pygame.time.get_ticks()
tick_counter   = 0

running = True
while running:
    dt = clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    now = pygame.time.get_ticks()
    if now - last_step_time >= TICK_MS:
        game_step()
        last_step_time = now
        tick_counter  += 1

    # ── Draw ──
    screen.fill(C_BG)
    draw_grid(screen)
    draw_food(screen, tick_counter)
    draw_snake(screen)
    draw_panel(screen)

    pygame.display.flip()

pygame.quit()
sys.exit()