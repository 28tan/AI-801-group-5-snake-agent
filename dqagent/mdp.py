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

    def get_state(self, game):
        return (
            tuple(game.snake),
            game.direction,
            game.food,
            game.done,
        )

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

        return self.get_state(game), reward, game.done


