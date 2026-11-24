
import argparse
from pathlib import Path

import pygame
import torch

from snake import SnakeGame
from agent import DQNAgent


MODEL_FILE = (
    Path(__file__).resolve().parent
    / "model"
    / "model.pth"
)


def main(untrained=False):
    pygame.init()

    game = SnakeGame()
    agent = DQNAgent()

    # Load the trained DQN model.
    if not untrained:
        if not MODEL_FILE.exists():
            pygame.quit()
            raise SystemExit(
                "No trained model found. "
                "Run python train.py first."
            )

        weights = torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=True
        )

        agent.model.load_state_dict(weights)
        print("Loaded trained DQN model.")

    else:
        print("Running an untrained DQN agent.")

    # Only predict actions, do not train.
    agent.model.eval()

    screen = pygame.display.set_mode((600, 440))
    pygame.display.set_caption(
        "Snake Game - Deep Q-Learning Agent"
    )

    font = pygame.font.SysFont(
        "arial",
        26,
        bold=True
    )

    small_font = pygame.font.SysFont(
        "arial",
        17
    )

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r and game.done:
                    game.reset()

        # DQN agent controls the snake automatically.
        if running and not game.done:
            state = agent.mdp.get_state(game)

            final_move, action = agent.get_action(
                state,
                training=False
            )

            next_state, reward, done = (
                agent.mdp.transition(game, action)
            )

        # Draw your existing Snake game.
        game.draw(screen, font)

        if game.done:
            overlay = pygame.Surface(
                (600, 70),
                pygame.SRCALPHA
            )

            overlay.fill((0, 0, 0, 190))
            screen.blit(overlay, (0, 165))

            message = font.render(
                "Game Over! Press R to restart",
                True,
                (255, 255, 255)
            )

            screen.blit(message, (85, 185))

        hint = small_font.render(
            "R: Restart   ESC: Quit",
            True,
            (190, 200, 215)
        )

        screen.blit(hint, (10, 412))

        pygame.display.flip()
        clock.tick(8)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--untrained",
        action="store_true",
        help="Watch the DQN before training"
    )

    args = parser.parse_args()
    main(untrained=args.untrained)
