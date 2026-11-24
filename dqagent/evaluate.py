
import argparse
import csv
import random
from pathlib import Path

import pygame
import torch

from snake import SnakeGame
from agent import DQNAgent


BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "model" / "model.pth"
RESULTS = BASE / "results" / "dqn_evaluation.csv"


def evaluate(games=100, seed=42, fps=30, show=True):
    if not MODEL_FILE.exists():
        raise SystemExit(
            "Model not found. Train first using python train.py"
        )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    game = SnakeGame()
    agent = DQNAgent()

    # Load the trained neural network.
    weights = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=True
    )

    agent.model.load_state_dict(weights)
    agent.model.eval()

    scores = []

    screen = None
    clock = None

    if show:
        pygame.init()
        screen = pygame.display.set_mode((600, 440))
        pygame.display.set_caption("DQN Agent - Evaluation")
        font = pygame.font.SysFont("arial", 26, bold=True)
        small_font = pygame.font.SysFont("arial", 17)
        clock = pygame.time.Clock()

    running = True

    try:
        with open(RESULTS, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow([
                "game", "score", "steps",
                "total_reward", "outcome"
            ])

            for game_number in range(1, games + 1):
                if not running:
                    break

                random.seed(seed + game_number)
                game.reset()

                steps = 0
                steps_without_food = 0
                total_reward = 0.0
                outcome = "collision"

                while not game.done and running:

                    if show:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    running = False

                    if not running:
                        break

                    previous_score = game.score

                    # Get the current MDP state.
                    state = agent.mdp.get_state(game)

                    # Select an action using the trained DQN.
                    # No training or random exploration.
                    final_move, action = agent.get_action(
                        state,
                        training=False
                    )

                    _, reward, done = agent.mdp.transition(
                        game, action
                    )

                    steps += 1

                    if game.score > previous_score:
                        steps_without_food = 0
                    else:
                        steps_without_food += 1

                    # Use the same timeout as MDP evaluation.
                    if (
                        not done
                        and steps_without_food > 100 * len(game.snake)
                    ):
                        game.done = True
                        done = True
                        reward = agent.mdp.DEATH_REWARD
                        outcome = "timeout"

                    elif done and game.food is None:
                        outcome = "board_completed"

                    total_reward += reward

                    if show:
                        # Draw the same Snake game.
                        game.draw(screen, font)

                        hint = small_font.render(
                            f"DQN Game: {game_number}/{games}   "
                            "ESC: Quit",
                            True,
                            (190, 200, 215)
                        )
                        screen.blit(hint, (10, 412))

                        if game.done:
                            overlay = pygame.Surface(
                                (600, 70), pygame.SRCALPHA
                            )
                            overlay.fill((0, 0, 0, 190))
                            screen.blit(overlay, (0, 165))

                            message = font.render(
                                f"Game Over! Score: {game.score}",
                                True,
                                (255, 255, 255)
                            )
                            screen.blit(message, (115, 185))

                        pygame.display.flip()
                        clock.tick(fps)

                # Do not record an interrupted game.
                if not running:
                    break

                scores.append(game.score)

                writer.writerow([
                    game_number,
                    game.score,
                    steps,
                    round(total_reward, 4),
                    outcome
                ])
                file.flush()

                print(
                    f"DQN Game {game_number}/{games} | "
                    f"Score: {game.score} | "
                    f"Steps: {steps} | "
                    f"Reward: {total_reward:.2f} | "
                    f"Outcome: {outcome}"
                )

                if show:
                    pygame.time.wait(250)

    finally:
        if show:
            pygame.quit()

    if scores:
        print("\nDQN evaluation finished.")
        print(f"Games completed: {len(scores)}")
        print(f"Average score: {sum(scores) / len(scores):.2f}")
        print(f"Highest score: {max(scores)}")
        print(f"Results saved to: {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-display", action="store_true")

    args = parser.parse_args()

    evaluate(
        games=args.games,
        seed=args.seed,
        fps=args.fps,
        show=not args.no_display
    )
