import pygame


def show_mdp_page():
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Snake MDP Explanation")

    background = (18, 24, 32)
    panel = (32, 43, 56)
    white = (240, 244, 248)
    green = (90, 230, 150)
    blue = (100, 180, 255)
    yellow = (255, 210, 90)
    gray = (185, 195, 210)

    title_font = pygame.font.SysFont("arial", 30, bold=True)
    heading_font = pygame.font.SysFont("arial", 21, bold=True)
    font = pygame.font.SysFont("arial", 17)

    clock = pygame.time.Clock()
    running = True

    def draw_text(text, x, y, color=white, font=font):
        image = font.render(text, True, color)
        screen.blit(image, (x, y))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_m):
                    return True

        screen.fill(background)

        draw_text(
            "SNAKE: MARKOV DECISION PROCESS",
            24, 20, green, title_font
        )

        # MDP definition
        pygame.draw.rect(
            screen, panel, (20, 75, 760, 65),
            border_radius=10
        )
        draw_text(
            "MDP = (S, A, P, R)",
            40, 88, yellow, heading_font
        )
        draw_text(
            "State, actions, transitions and rewards",
            40, 115
        )

        # State
        pygame.draw.rect(
            screen, panel, (20, 155, 760, 105),
            border_radius=10
        )
        draw_text("1. STATE (S)", 40, 170, blue, heading_font)
        draw_text(
            "Snake body, current direction and food position",
            40, 202
        )
        draw_text(
            "Terminal flag indicates whether the game ended.",
            40, 228
        )

        # Actions
        pygame.draw.rect(
            screen, panel, (20, 275, 365, 120),
            border_radius=10
        )
        draw_text("2. ACTIONS (A)", 40, 290, blue, heading_font)
        draw_text("UP, DOWN, LEFT, RIGHT", 40, 326)
        draw_text("The agent chooses a legal move.", 40, 355)

        # Transition
        pygame.draw.rect(
            screen, panel, (405, 275, 375, 120),
            border_radius=10
        )
        draw_text("3. TRANSITION (P)", 425, 290, blue, heading_font)
        draw_text("Apply action to the game.", 425, 326)
        draw_text("Update body, direction and food.", 425, 355)

        # Rewards
        pygame.draw.rect(
            screen, panel, (20, 410, 365, 130),
            border_radius=10
        )
        draw_text("4. REWARDS (R)", 40, 425, blue, heading_font)
        draw_text("+10: food eaten", 40, 462, green)
        draw_text("-10: death", 40, 487, (255, 110, 110))
        draw_text("-0.1: ordinary move", 40, 512, yellow)

        # Policy
        pygame.draw.rect(
            screen, panel, (405, 410, 375, 130),
            border_radius=10
        )
        draw_text("5. AGENT POLICY", 425, 425, blue, heading_font)
        draw_text("Avoid immediate collisions.", 425, 462)
        draw_text("Prefer moves toward the food.", 425, 487)
        draw_text("No learning or training.", 425, 512, green)

        draw_text(
            "Press M or ESC to return to the game.",
            24, 565, gray
        )

        pygame.display.flip()
        clock.tick(30)

    return False