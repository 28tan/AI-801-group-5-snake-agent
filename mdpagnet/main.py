import pygame

from snake import SnakeGame
from mdp import SnakeMDP, MDPAgent


def main():
    pygame.init()

    game = SnakeGame()
    mdp = SnakeMDP()
    agent = MDPAgent(mdp)

    screen = pygame.display.set_mode((600, 440))
    pygame.display.set_caption("Snake Game - MDP Agent")

    font = pygame.font.SysFont("arial", 26, bold=True)
    small_font = pygame.font.SysFont("arial", 17)
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

        # Agent controls the snake automatically
        if running and not game.done:
            action = agent.choose_action(game)
            next_state, reward, done = mdp.transition(game, action)

        # Draw the game
        game.draw(screen, font)

        if game.done:
            overlay = pygame.Surface((600, 70), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 190))
            screen.blit(overlay, (0, 165))

            message = font.render(
                "Game Over! Press R to restart",
                True,
                (255, 255, 255),
            )
            screen.blit(message, (85, 185))

        hint = small_font.render(
            "R: Restart   ESC: Quit",
            True,
            (190, 200, 215),
        )
        screen.blit(hint, (10, 412))

        pygame.display.flip()
        clock.tick(8)

    pygame.quit()


if __name__ == "__main__":
    main()