import turtle
import random
import torch
import collections

screen = turtle.Screen()
screen.setup(1050, 650)
screen.bgcolor("black")
screen.title("Snake AI - DQN")
screen.tracer(0)

atminties_buferis = collections.deque(maxlen=100_000)

snake = turtle.Turtle()
snake.speed(0)
snake.shape("square")
snake.color("white")
snake.penup()
snake.resizemode("noresize")
snake.goto(10, 10)
snake.setheading(0)

game_running = True
score = 0
moves = 0
atlygis = 0
body = []
deaths = 0
episode = 0
last_loss = 0.0
best_score = 0
epsilon = 0.9

class SnakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sluoksnis1 = torch.nn.Linear(11, 16)
        self.sluoksnis2 = torch.nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.sluoksnis1(x))
        x = self.sluoksnis2(x)
        return x
    
tinklas = SnakeNet()
optimizer = torch.optim.Adam(tinklas.parameters(), lr=0.001)

def learning():
    global last_loss
    if len(atminties_buferis) < 32:
        return
    partija = random.sample(atminties_buferis, 32)
    busenos, veiksmai, atlygis, naujos_busenos = zip(*partija)

    busenos = torch.FloatTensor(busenos)
    veiksmai = torch.LongTensor(veiksmai)
    atlygis = torch.FloatTensor(atlygis)
    naujos_busenos = torch.FloatTensor(naujos_busenos)

    rezultatas = tinklas(busenos)
    q_dabartinis = rezultatas.gather(1, veiksmai.unsqueeze(1))

    #tikslas = atlygis + 0.9 × geriausias_kitas_žingsnis
    q_kitas = tinklas(naujos_busenos)
    q_tikslas = (q_kitas.max(1)[0] * 0.9 + atlygis).detach()
    klaida = torch.nn.functional.mse_loss(q_dabartinis, q_tikslas.unsqueeze(1))
    optimizer.zero_grad()
    klaida.backward()
    optimizer.step()
    last_loss = klaida.item()


GAME_LEFT   = -280
GAME_RIGHT  = 200
GAME_BOTTOM = -220
GAME_TOP    = 200
PANEL_X     = 230   # left edge of stats panel text

def grid():
    grid_turtle = turtle.Turtle()
    grid_turtle.speed(0)
    grid_turtle.color("#00FF41")
    grid_turtle.penup()
    for x in range(-220, 220, 20):
        grid_turtle.goto(x, -220)
        grid_turtle.pendown()
        grid_turtle.goto(x, 200)
        grid_turtle.penup()
    for y in range(-220, 220, 20):
        grid_turtle.goto(-220, y)
        grid_turtle.pendown()
        grid_turtle.goto(200, y)
        grid_turtle.penup()
    grid_turtle.hideturtle()

def draw_divider():
    div = turtle.Turtle()
    div.speed(0)
    div.color("#333333")
    div.penup()
    div.goto(215, 280)
    div.pendown()
    div.goto(215, -280)
    div.penup()
    div.hideturtle()

# top bar for score/moves
stats_turtle = turtle.Turtle()
stats_turtle.speed(0)
stats_turtle.color("white")
stats_turtle.penup()
stats_turtle.goto(0, 215)
stats_turtle.hideturtle()

# right-side panel turtle
panel_turtle = turtle.Turtle()
panel_turtle.speed(0)
panel_turtle.penup()
panel_turtle.hideturtle()

def _net_param_count(model):
    return sum(p.numel() for p in model.parameters())

def game_stats():

    # top bar
    stats_turtle.clear()
    stats_turtle.color("#00FF41")
    stats_turtle.write(
        f"Score: {score}   Best: {best_score}   Moves: {moves}",
        align="center", font=("Courier", 13, "bold")
    )

    # right panel
    panel_turtle.clear()

    title_x = PANEL_X + 60
    col_x   = PANEL_X

    def row(y, label, value, color="white"):
        panel_turtle.color("#888888")
        panel_turtle.goto(col_x, y)
        panel_turtle.write(label, align="left", font=("Courier", 10, "normal"))
        panel_turtle.color(color)
        panel_turtle.goto(col_x + 130, y)
        panel_turtle.write(str(value), align="left", font=("Courier", 10, "bold"))

    # --- header ---
    panel_turtle.color("#00FF41")
    panel_turtle.goto(title_x, 215)
    panel_turtle.write("STATS", align="center", font=("Courier", 13, "bold"))

    panel_turtle.color("#444444")
    panel_turtle.goto(col_x, 200)
    panel_turtle.write("─" * 22, align="left", font=("Courier", 10, "normal"))

    # --- game stats ---
    panel_turtle.color("#aaaaaa")
    panel_turtle.goto(col_x, 182)
    panel_turtle.write("GAME", align="left", font=("Courier", 9, "bold"))

    row(164, "Score",       score,          "#00FF41")
    row(148, "Best Score",  best_score,     "#FFD700")
    row(132, "Episode",     episode,        "white")
    row(116, "Deaths",      deaths,         "#FF4444")
    row(100, "Moves",       moves,          "white")

    panel_turtle.color("#444444")
    panel_turtle.goto(col_x, 86)
    panel_turtle.write("─" * 22, align="left", font=("Courier", 10, "normal"))

    # --- RL stats ---
    panel_turtle.color("#aaaaaa")
    panel_turtle.goto(col_x, 68)
    panel_turtle.write("RL / TRAINING", align="left", font=("Courier", 9, "bold"))

    row(50,  "Epsilon",     f"{epsilon:.2f}",             "#FFD700")
    row(34,  "Buffer",      f"{len(atminties_buferis):,}", "white")
    row(18,  "Buf. Max",    f"{atminties_buferis.maxlen:,}", "#888888")
    row(2,   "Batch",       "32",                          "#888888")
    row(-14, "Gamma",       "0.9",                         "#888888")
    row(-30, "Loss",        f"{last_loss:.4f}",            "#FF9944")

    panel_turtle.color("#444444")
    panel_turtle.goto(col_x, -44)
    panel_turtle.write("─" * 22, align="left", font=("Courier", 10, "normal"))

    # --- network params ---
    panel_turtle.color("#aaaaaa")
    panel_turtle.goto(col_x, -62)
    panel_turtle.write("NETWORK", align="left", font=("Courier", 9, "bold"))

    row(-80,  "Architecture", "11→16→3",                      "#88CCFF")
    row(-96,  "Layer 1",      "Linear(11, 16)",                "#88CCFF")
    row(-112, "Layer 2",      "Linear(16, 3)",                 "#88CCFF")
    row(-128, "Activation",   "ReLU",                          "#88CCFF")
    row(-144, "Params",       f"{_net_param_count(tinklas):,}", "white")
    row(-160, "Optimizer",    "Adam",                          "#88CCFF")
    row(-176, "LR",           "0.001",                         "#88CCFF")
    row(-192, "Inputs",       "11 (food×4, danger×3, dir×4)", "#aaaaaa")
    row(-208, "Outputs",      "4 (U/R/D/L)",                   "#aaaaaa")


def boundaries():
    snake_x, snake_y = snake.position()
    global game_running
    if snake_x < -210 or snake_x > 190 or snake_y < -210 or snake_y > 190:
        game_running = False
    if snake.position() in [part.position() for part in body]:
        game_running = False

current_food = turtle.Turtle()
current_food.speed(0)
current_food.shape("square")
current_food.color("red")
current_food.penup()
current_food.resizemode("noresize")

def place_food():
    while True:
        x = random.randint(-11, 9) * 20
        y = random.randint(-11, 9) * 20
        food_x = x + 10
        food_y = y + 10
        if (food_x, food_y) not in [part.position() for part in body]:
            current_food.goto(food_x, food_y)
            return

def get_state(head_x, head_y):
    body_positions = [part.position() for part in body]
    food_x, food_y = current_food.position()
    heading = snake.heading()

    def is_blocked(nx, ny):
        return nx < -210 or nx > 190 or ny < -210 or ny > 190 or (nx, ny) in body_positions

    if heading == 0:        # dešinėn
        ar_kliutis_priekis = is_blocked(head_x + 20, head_y)
        ar_kliutis_kaire   = is_blocked(head_x, head_y + 20)
        ar_kliutis_desine  = is_blocked(head_x, head_y - 20)
    elif heading == 90:     # aukštyn
        ar_kliutis_priekis = is_blocked(head_x, head_y + 20)
        ar_kliutis_kaire   = is_blocked(head_x - 20, head_y)
        ar_kliutis_desine  = is_blocked(head_x + 20, head_y)
    elif heading == 180:    # kairėn
        ar_kliutis_priekis = is_blocked(head_x - 20, head_y)
        ar_kliutis_kaire   = is_blocked(head_x, head_y - 20)
        ar_kliutis_desine  = is_blocked(head_x, head_y + 20)
    elif heading == 270:    # žemyn
        ar_kliutis_priekis = is_blocked(head_x, head_y - 20)
        ar_kliutis_kaire   = is_blocked(head_x + 20, head_y)
        ar_kliutis_desine  = is_blocked(head_x - 20, head_y)

    return [
        int(food_x > head_x),
        int(food_x < head_x),
        int(food_y > head_y),
        int(food_y < head_y),
        int(ar_kliutis_priekis),
        int(ar_kliutis_kaire),
        int(ar_kliutis_desine),
        int(heading == 0),
        int(heading == 90),
        int(heading == 180),
        int(heading == 270),
    ]

def atmintis(buvusi_busena):
    global atlygis
    veiksmas          = snake.heading() // 90
    dabartinis_atlygis = atlygis
    nauja_busena      = get_state(*snake.position())
    atminties_buferis.append((buvusi_busena, veiksmas, dabartinis_atlygis, nauja_busena))
    atlygis = 0

def judesio_pasirinkimas():
    busena = get_state(*snake.position())
    q_reiksmes = tinklas(torch.FloatTensor([busena]))

    if random.random() < epsilon:
        veiksmas = random.randint(0, 3)
    else:
        veiksmas = q_reiksmes.argmax().item()

    dabartine_kryptis = snake.heading()
    nauja_kryptis = veiksmas * 90
    if abs(dabartine_kryptis - nauja_kryptis) == 180:
        pass
    else:
        snake.setheading(nauja_kryptis)

def reset():
    global score, moves, atlygis, game_running, deaths, episode, best_score, epsilon
    if score > best_score:
        best_score = score
    deaths += 1
    episode += 1
    epsilon = max(0.05, epsilon * 0.995)
    score = 0
    moves = 0
    atlygis = 0
    game_running = True
    snake.goto(10, 10)
    snake.setheading(0)
    for part in body:
        part.goto(1000, 1000)
    body.clear()
    place_food()
    snake_move()

def snake_move():
    global score, game_running, moves, atlygis

    if game_running:
        moves += 1
        game_stats()
        judesio_pasirinkimas()
        head_x, head_y = snake.position()
        heading = snake.heading()

        if heading == 0:
            next_x, next_y = head_x + 20, head_y
        elif heading == 180:
            next_x, next_y = head_x - 20, head_y
        elif heading == 90:
            next_x, next_y = head_x, head_y + 20
        elif heading == 270:
            next_x, next_y = head_x, head_y - 20
        else:
            import math
            rad = math.radians(heading)
            next_x = round(head_x + 20 * math.cos(rad))
            next_y = round(head_y + 20 * math.sin(rad))

        # Užfiksuojam būseną PRIEŠ žingsnį
        busena_pries = get_state(*snake.position())

        # Tikrinam mirtį
        if next_x < -210 or next_x > 190 or next_y < -210 or next_y > 190:
            game_running = False
            atlygis -= 10
            atmintis(busena_pries)
            learning()
            reset()
            return

        if (next_x, next_y) in [part.position() for part in body]:
            game_running = False
            atlygis -= 10
            atmintis(busena_pries)
            learning()
            reset()
            return

        # Judinam kūną
        snake_coordinates = snake.position()
        for i in range(len(body) - 1, 0, -1):
            body[i].goto(body[i-1].position())
        if len(body) > 0:
            body[0].goto(snake_coordinates)

        # Judinam galvą
        snake.goto(next_x, next_y)

        # Išsaugom žingsnį į buferį
        atmintis(busena_pries)
        learning()

        # Maisto patikrinimas
        if snake.distance(current_food) < 5:
            score += 1
            atlygis += 10
            place_food()

            new_body_part = turtle.Turtle()
            new_body_part.hideturtle()
            new_body_part.speed(0)
            new_body_part.shape("square")
            new_body_part.color("white")
            new_body_part.penup()
            new_body_part.resizemode("noresize")
            new_body_part.goto(body[-1].position() if body else snake_coordinates)
            new_body_part.showturtle()
            body.append(new_body_part)

        boundaries()
        screen.update()
        screen.ontimer(snake_move, 15)

screen.onkey(lambda: snake.setheading(90),  'Up')
screen.onkey(lambda: snake.setheading(180), 'Left')
screen.onkey(lambda: snake.setheading(0),   'Right')
screen.onkey(lambda: snake.setheading(270), 'Down')
screen.listen()

grid()
draw_divider()
place_food()
snake_move()
screen.mainloop()