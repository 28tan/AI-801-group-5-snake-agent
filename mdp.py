from snake import get_state


class SnakeMDP:
    """
    Full-state MDP for Snake.

    State: snake body, direction, food, terminal flag
    Actions: UP, DOWN, LEFT, RIGHT
    Rewards: +10 food, -10 death, -0.1 normal move
    """

    ACTIONS = {
        "UP": (0, -20),
        "DOWN": (0, 20),
        "LEFT": (-20, 0),
        "RIGHT": (20, 0),
    }

    OPPOSITE = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT",
    }

    FOOD_REWARD = 10
    DEATH_REWARD = -10
    STEP_REWARD = -0.1

    def transition(self, game, action):
        """Apply an action and return (next_state, reward, done)."""
        old_score = game.score

        game.step(self.ACTIONS[action])

        if game.score > old_score:
            reward = self.FOOD_REWARD
        elif game.done:
            reward = self.DEATH_REWARD
        else:
            reward = self.STEP_REWARD

        return get_state(game), reward, game.done


class MDPAgent:
    """
    Greedy, rule-based policy. No learning or training.
    """

    def __init__(self, mdp):
        self.mdp = mdp

    def choose_action(self, game):
        hx, hy = game.snake[0]
        fx, fy = game.food

        current_direction = game.direction

        current_action = next(
            name
            for name, vector in self.mdp.ACTIONS.items()
            if vector == current_direction
            if vector == current_direction
        )

        candidates = sorted(
            self.mdp.ACTIONS,
            key=lambda action: (
                abs(hx + self.mdp.ACTIONS[action][0] - fx)
                + abs(hy + self.mdp.ACTIONS[action][1] - fy)
            ),
        )

        for action in candidates:
            # Never reverse directly into the neck.
            if action == self.mdp.OPPOSITE[current_action]:
                continue

            dx, dy = self.mdp.ACTIONS[action]
            nx, ny = hx + dx, hy + dy
            new_head = (nx, ny)

            # Reject moves outside the board.
            if not (
                0 <= nx < game.WIDTH
                and 0 <= ny < game.HEIGHT
            ):
                continue

            # The tail vacates its cell if food is not eaten.
            body = (
                game.snake
                if new_head == game.food
                else game.snake[:-1]
            )

            # Reject immediate self-collisions.
            if new_head in body:
                continue

            return action

        # If no safe option exists, choose straight ahead.
        return current_action