"""Evaluate each saved DQN checkpoint on the same set of Snake games."""

import argparse
import csv
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import pygame
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dqagent"))

from snake import SnakeGame
from agent import DQNAgent, to_features
from model import Linear_QNet


MODELS = ROOT / "finalResults" / "models"
EVALUATIONS = ROOT / "finalResults" / "evaluations"


def load_model(model_file):
    weights = torch.load(
        model_file,
        map_location="cpu",
        weights_only=True
    )

    input_size = weights["linear1.weight"].shape[1]
    hidden_size = weights["linear1.weight"].shape[0]

    model = Linear_QNet(input_size, hidden_size, 3)
    model.load_state_dict(weights)
    model.eval()
    return model, input_size, hidden_size


def get_action(agent, state, input_size):
    state0 = torch.tensor(
        to_features(state)[:input_size],
        dtype=torch.float32
    )

    with torch.no_grad():
        prediction = agent.model(state0)

    move = int(torch.argmax(prediction).item())

    # Convert relative turn into MDP direction.
    _, (dx, dy), _, _ = state
    if move == 1: # Turn right
        dx, dy = -dy, dx
    elif move == 2: # Turn left
        dx, dy = dy, -dx
    action = next(name for name, delta in agent.mdp.ACTIONS.items()
        if tuple(delta) == (dx, dy))
    return action


def evaluate(model_file, results_file, games=100, seed=42, fps=30, show=True):
    game = SnakeGame()
    agent = DQNAgent()

    # Load the trained neural network.
    agent.model, input_size, hidden_size = load_model(model_file)

    scores = []

    screen = None
    clock = None

    if show:
        pygame.init()
        screen = pygame.display.set_mode((600, 440))
        pygame.display.set_caption(f"DQN Agent - {model_file.stem}")
        font = pygame.font.SysFont("arial", 26, bold=True)
        small_font = pygame.font.SysFont("arial", 17)
        clock = pygame.time.Clock()

    running = True

    try:
        with open(results_file, "w", newline="", encoding="utf-8") as file:
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
                    action = get_action(agent, state, input_size)

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
                            f"{model_file.stem}  Game: {game_number}/{games}   "
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

                if show:
                    pygame.time.wait(250)

    finally:
        if show:
            pygame.quit()

    if not running:
        return None

    print(
        f"{model_file.stem} | "
        f"Average score: {sum(scores) / len(scores):.2f} | "
        f"Highest score: {max(scores)}"
    )

    return {
        "agent": model_file.stem,
        "features": input_size,
        "hidden": hidden_size,
        "games": games,
        "average_score": round(statistics.mean(scores), 2),
        "median_score": statistics.median(scores),
        "best_score": max(scores),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-display", action="store_true")

    args = parser.parse_args()

    model_files = sorted(MODELS.glob("*.pth"))
    if not model_files:
        raise SystemExit("No model checkpoints found in finalResults/models")

    output = EVALUATIONS / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)

    for model_file in model_files:
        result = evaluate(
            model_file,
            output / f"evaluation_{model_file.stem}.csv",
            games=args.games,
            seed=args.seed,
            fps=args.fps,
            show=not args.no_display
        )
        if result is None:
            raise SystemExit(f"Evaluation stopped. Partial results are in {output}")

    print(f"\nResults saved to: {output}")
