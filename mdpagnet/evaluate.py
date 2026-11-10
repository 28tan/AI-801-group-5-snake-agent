
import argparse
import csv
import random
from pathlib import Path

import pygame

from snake import SnakeGame
from mdp import SnakeMDP, MDPAgent


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results" / "mdp_evaluation.csv"


def evaluate(games=100, seed=42, fps=30, show=True):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    game = SnakeGame()
    mdp = SnakeMDP()
    agent = MDPAgent(mdp)

    scores = []

    screen = None
    clock = None

    if show:
        pygame.init()
        screen = pygame.display.set_mode((600, 440))
        pygame.display.set_caption("MDP Agent - Evaluation")
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

                    # MDP agent chooses its next action.
                    action = agent.choose_action(game)

                    _, reward, done = mdp.transition(
                        game, action
                    )

                    steps += 1

                    if game.score > previous_score:
                        steps_without_food = 0
                    else:
                        steps_without_food += 1

                    # Stop endless loops without food.
                    if (
                        not done
                        and steps_without_food > 100 * len(game.snake)
                    ):
                        game.done = True
                        done = True
                        reward = mdp.DEATH_REWARD
                        outcome = "timeout"

                    elif done and game.food is None:
                        outcome = "board_completed"

                    total_reward += reward

                    if show:
                        # Draw the Snake game.
                        game.draw(screen, font)

                        hint = small_font.render(
                            f"MDP Game: {game_number}/{games}   "
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
                    f"MDP Game {game_number}/{games} | "
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
        print("\nMDP evaluation finished.")
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
