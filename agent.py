import torch
import random
import numpy as np
from collections import deque

from snake import SnakeGame, get_state
from mdp import SnakeMDP
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


def to_features(state):
    """Turn the MDP state (snake, direction, food, done) into the network's 11 values."""
    snake, (dx, dy), food, _ = state
    hx, hy = snake[0]
    body = {tuple(segment) for segment in snake[1:-1]}  # the tail moves away this step

    def danger(x, y):
        # off the board or into the body
        return x < 0 or x >= SnakeGame.WIDTH or y < 0 or y >= SnakeGame.HEIGHT or (x, y) in body

    right, left = (-dy, dx), (dy, -dx)  # turning right/left on screen (y grows downward)
    features = [
        danger(hx + dx, hy + dy),  # danger straight
        danger(hx + right[0], hy + right[1]),  # danger right
        danger(hx + left[0], hy + left[1]),  # danger left
        dx < 0, dx > 0, dy < 0, dy > 0,  # moving left, right, up, down
    ]
    if food is None:  # board full: no food left
        features += [False] * 4
    else:
        fx, fy = food
        features += [fx < hx, fx > hx, fy < hy, fy > hy]  # food left, right, up, down
    return np.array(features, dtype=int)


class DQNAgent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.mdp = SnakeMDP()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(np.array(states), actions, rewards, np.array(next_states), dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(to_features(state), dtype=torch.float)
            move = torch.argmax(self.model(state0)).item()
        final_move = [0, 0, 0]
        final_move[move] = 1  # [straight, right, left], for training

        # turn the move into our MDP's action name ("UP", "DOWN", "LEFT", "RIGHT")
        _, (dx, dy), _, _ = state
        if move == 1:  # right turn
            dx, dy = -dy, dx
        elif move == 2:  # left turn
            dx, dy = dy, -dx
        action = next(name for name, delta in SnakeMDP.ACTIONS.items()
                      if tuple(delta) == (dx, dy))
        return final_move, action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = DQNAgent()
    game = SnakeGame()
    steps_since_food = 0
    while True:
        state_old = get_state(game)
        features_old = to_features(state_old)  # capture before game.step mutates the snake

        final_move, action = agent.get_action(state_old)

        state_new, reward, done = agent.mdp.transition(game, action)
        score = game.score

        steps_since_food = 0 if reward == agent.mdp.FOOD_REWARD else steps_since_food + 1
        if not done and steps_since_food > 100 * len(game.snake):
            done = True
            reward = agent.mdp.DEATH_REWARD

        features_new = to_features(state_new)

        agent.train_short_memory(features_old, final_move, reward, features_new, done)

        agent.remember(features_old, final_move, reward, features_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            steps_since_food = 0
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
