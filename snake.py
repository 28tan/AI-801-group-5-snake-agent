import random
import pygame


CELL = 20
WIDTH = 600
HEIGHT = 400


class SnakeGame:
    WIDTH = WIDTH
    HEIGHT = HEIGHT
    CELL = CELL

    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [
            (200, 200),
            (180, 200),
            (160, 200),
        ]
        self.direction = (CELL, 0)
        self.score = 0
        self.done = False
        self.food = self.spawn_food()

    def spawn_food(self):
        empty_cells = [
            (x, y)
            for x in range(0, WIDTH, CELL)
            for y in range(0, HEIGHT, CELL)
            if (x, y) not in self.snake
        ]

        return random.choice(empty_cells) if empty_cells else None

    def step(self, direction=None):
        if self.done:
            return

        if direction is not None:
            dx, dy = direction
            reverse = (-self.direction[0], -self.direction[1])

            if (dx, dy) != reverse:
                self.direction = (dx, dy)

        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        # Check whether the snake eats food.
        eating = new_head == self.food

        # The tail moves away on a normal move.
        collision_body = (
            self.snake if eating else self.snake[:-1]
        )

        # Check wall and self collisions.
        x, y = new_head

        if (
            x < 0 or x >= WIDTH
            or y < 0 or y >= HEIGHT
            or new_head in collision_body
        ):
            self.done = True
            return

        self.snake.insert(0, new_head)

        if eating:
            self.score += 1
            self.food = self.spawn_food()

            if self.food is None:
                self.done = True
        else:
            self.snake.pop()

    def draw(self, screen, font):
        screen.fill((18, 24, 32))

        # Draw food.
        if self.food is not None:
            pygame.draw.rect(
                screen,
                (255, 85, 90),
                (*self.food, CELL, CELL),
                border_radius=5,
            )

        # Draw snake.
        for index, segment in enumerate(self.snake):
            color = (
                (90, 240, 145)
                if index == 0
                else (35, 170, 100)
            )

            pygame.draw.rect(
                screen,
                color,
                (
                    segment[0] + 1,
                    segment[1] + 1,
                    CELL - 2,
                    CELL - 2,
                ),
                border_radius=4,
            )

        score_text = font.render(
            f"Score: {self.score}",
            True,
            (245, 245, 245),
        )
        screen.blit(score_text, (10, 8))


def get_state(self):
    return (
        self.snake,
        self.direction,
        self.food,
        self.done,
    )