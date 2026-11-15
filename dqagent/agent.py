
import heapq
import torch
import random
import numpy as np
from collections import deque

from snake import SnakeGame
from mdp import SnakeMDP
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

def has_path_to_food_greedy(snake, food):
    if food is None:
        return False

    cols = SnakeGame.WIDTH // SnakeGame.CELL
    rows = SnakeGame.HEIGHT // SnakeGame.CELL

    def to_cell(point):
        return point[0] // SnakeGame.CELL, point[1] // SnakeGame.CELL

    grid = [[False] * cols for _ in range(rows)]

    for segment in snake:
        cx, cy = to_cell(segment)
        if 0 <= cx < cols and 0 <= cy < rows:
            grid[cy][cx] = True

    start = to_cell(snake[0])
    gx, gy = to_cell(food)

    visited = {start}
    frontier = [(0, start)]

    while frontier:
        _, (cx, cy) = heapq.heappop(frontier)

        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if (nx, ny) == (gx, gy):
                return True

            if (
                0 <= nx < cols
                and 0 <= ny < rows
                and not grid[ny][nx]
                and (nx, ny) not in visited
            ):
                visited.add((nx, ny))
                distance = abs(nx - gx) + abs(ny - gy)
                heapq.heappush(frontier, (distance, (nx, ny)))

    return False


def to_features(state):

    snake, (dx, dy), food, done = state
    hx, hy = snake[0]

    def danger(x, y):
        new_head = (x, y)

        if new_head == food:
            # The snake grows when eating food.
            # Its tail stays, so check the entire body.
            body = snake[1:]
        else:
            # During a normal move the tail moves away.
            # Exclude the tail from collision detection.
            body = snake[1:-1]

        return (
            x < 0
            or x >= SnakeGame.WIDTH
            or y < 0
            or y >= SnakeGame.HEIGHT
            or new_head in body
        )

    def line_of_sight(snake, food):
        if food is None:
            return False

        hx, hy = snake[0]
        fx, fy = food

        # Food must be in the same row or column as the head.
        if hx != fx and hy != fy:
            return False

        # Check whether a body segment blocks the straight path.
        for x, y in snake[1:]:

            # Horizontal line of sight.
            if hy == fy and y == hy:
                if min(hx, fx) < x < max(hx, fx):
                    return False

            # Vertical line of sight.
            if hx == fx and x == hx:
                if min(hy, fy) < y < max(hy, fy):
                    return False

        # No body segment blocks the straight path.
        return True

    # Right and left are relative to the current direction.
    right = (-dy, dx)
    left = (dy, -dx)

    features = [
        # Danger: straight, right, left
        danger(hx + dx, hy + dy),
        danger(hx + right[0], hy + right[1]),
        danger(hx + left[0], hy + left[1]),

        # Direction: left, right, up, down
        dx < 0,
        dx > 0,
        dy < 0,
        dy > 0,
    ]

    if food is None:
        features += [False] * 4
    else:
        fx, fy = food
        features += [
            fx < hx,
            fx > hx,
            fy < hy,
            fy > hy,
        ]
    los = line_of_sight(snake, food)
    features.append(los)
    if not los:
        features.append(has_path_to_food_greedy(snake, food))
    else:
        features.append(True)

    return np.array(features, dtype=np.float32)


class DQNAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(13, 256, 3)

        self.trainer = QTrainer(
            self.model,
            lr=LR,
            gamma=self.gamma
        )

        self.mdp = SnakeMDP()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def train_long_memory(self):
        if not self.memory:
            return

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory,
                BATCH_SIZE
            )
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(
            *mini_sample
        )

        # Keep your original method of passing
        # actions, rewards and dones directly.
        self.trainer.train_step(
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            dones
        )

    def train_short_memory(
        self, state, action, reward, next_state, done
    ):
        self.trainer.train_step(
            state, action, reward, next_state, done
        )

    def get_action(self, state, training=True):
        # Explore randomly during training.
        # Use the network during normal gameplay.
        self.epsilon = max(0, 80 - self.n_games)

        if training and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(
                to_features(state),
                dtype=torch.float32
            )

            with torch.no_grad():
                prediction = self.model(state0)

            move = int(torch.argmax(prediction).item())

        # DQN action: straight, right or left.
        final_move = [0, 0, 0]
        final_move[move] = 1

        # Convert relative turn into MDP direction.
        _, (dx, dy), _, _ = state
        if move == 1: # Turn right
            dx, dy = -dy, dx
        elif move == 2: # Turn left
            dx, dy = dy, -dx
        action = next(name for name, delta in self.mdp.ACTIONS.items()
            if tuple(delta) == (dx, dy))
        return final_move, action
