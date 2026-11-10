
import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from snake import SnakeGame
from agent import DQNAgent, to_features


BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model" / "model.pth"
RESULTS_DIR = BASE / "results"


def train(episodes=2000, log_file=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if log_file is None:
        log_file = RESULTS_DIR / "training_results.csv"
    else:
        log_file = Path(log_file)

        if not log_file.is_absolute():
            log_file = BASE / log_file

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    agent = DQNAgent()
    game = SnakeGame()

    record = 0
    total_score = 0
    recent_scores = deque(maxlen=100)

    # Record one row for each finished game.
    with open(
        log_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "score",
            "record",
            "mean_score",
            "mean_last_100",
            "steps",
            "episode_reward",
            "epsilon",
        ])

        for episode in range(1, episodes + 1):
            game.reset()

            steps_since_food = 0
            steps = 0
            episode_reward = 0.0

            # Exploration value for this episode.
            epsilon = max(0, 80 - agent.n_games)

            while not game.done:
                # 1. Observe full MDP state.
                state_old = agent.mdp.get_state(game)

                # 2. Choose a move.
                final_move, action = agent.get_action(
                    state_old,
                    training=True
                )

                # 3. Apply the move and get its reward.
                state_new, reward, done = (
                    agent.mdp.transition(game, action)
                )

                steps += 1

                if reward == agent.mdp.FOOD_REWARD:
                    steps_since_food = 0
                else:
                    steps_since_food += 1

                # Prevent endless loops without food.
                if (
                    not done
                    and steps_since_food > 100 * len(game.snake)
                ):
                    game.done = True
                    done = True
                    reward = agent.mdp.DEATH_REWARD

                    # Update the terminal state.
                    state_new = agent.mdp.get_state(game)

                episode_reward += reward

                # 4. Convert states into 11 features.
                features_old = to_features(state_old)
                features_new = to_features(state_new)

                # 5. Train on the latest experience.
                agent.train_short_memory(
                    features_old,
                    final_move,
                    reward,
                    features_new,
                    done
                )

                # 6. Store experience for replay.
                agent.remember(
                    features_old,
                    final_move,
                    reward,
                    features_new,
                    done
                )

            # One game has finished.
            agent.n_games += 1

            # Train using previous experiences.
            agent.train_long_memory()

            score = game.score
            total_score += score
            recent_scores.append(score)

            record = max(record, score)

            mean_score = total_score / episode
            mean_last_100 = (
                sum(recent_scores) / len(recent_scores)
            )

            # Save the episode's statistics to CSV.
            writer.writerow([
                episode,
                score,
                record,
                round(mean_score, 4),
                round(mean_last_100, 4),
                steps,
                round(episode_reward, 4),
                epsilon,
            ])

            file.flush()


            print(
                f"Episode {episode}/{episodes} | "
                f"Score: {score} | "
                f"Record: {record} | "
                f"Mean last 100: {mean_last_100:.2f}"
            )

    # Only one model file is saved.
    agent.model.save(MODEL_PATH)

    print("\nTraining finished.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Results saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--log",
        type=str,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        log_file=args.log,
        seed=args.seed
    )
