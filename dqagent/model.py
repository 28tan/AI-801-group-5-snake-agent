
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(
        self,
        input_size=11,
        hidden_size=256,
        output_size=3
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            input_size, hidden_size
        )
        self.linear2 = nn.Linear(
            hidden_size, output_size
        )

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, path=None):
        if path is None:
            path = (
                Path(__file__).resolve().parent
                / "model"
                / "model.pth"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.state_dict(),
            path
        )


class QTrainer:
    def __init__(self, model, lr=0.001, gamma=0.9):
        self.model = model
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones
    ):
        states = torch.as_tensor(
            states,
            dtype=torch.float32
        )
        actions = torch.as_tensor(
            actions,
            dtype=torch.long
        )
        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32
        )
        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32
        )

        # Add a batch dimension if training
        # on only one experience.
        if states.ndim == 1:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            dones = [dones]

        dones = torch.as_tensor(
            dones,
            dtype=torch.bool
        )

        # Predict Q-values for all three actions.
        predictions = self.model(states)

        # Find the action that was selected.
        action_indices = actions.argmax(dim=1)

        selected_q = predictions.gather(
            1,
            action_indices.unsqueeze(1)
        ).squeeze(1)

        # Compute Q-learning target.
        with torch.no_grad():
            future_q = self.model(next_states).max(
                dim=1
            ).values

            targets = (
                rewards
                + (~dones).float() * self.gamma * future_q
            )

        # Compare predictions with target Q-values.
        loss = self.criterion(
            selected_q,
            targets
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
