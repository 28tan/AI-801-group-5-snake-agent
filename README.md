# AI-801-group-5-snake-agent

mdp stuff first

# Snake AI Project — Commands

## 1. MDP Reward Agent

### Change to MDP folder

```powershell
cd D:\projects\pythonProjects\PythonProject\MDP_Reward
```

### Watch MDP play

```powershell
# Normal game (8 FPS)
python main.py
```

### Evaluate MDP and record results

```powershell
# Watch 10 games (default: 30 FPS)
python evaluate.py --games 10

# Watch 100 games
python evaluate.py --games 100

# Watch 100 games using seed 42
python evaluate.py --games 100 --seed 42

# Watch at 8 FPS (same as main.py)
python evaluate.py --games 100 --seed 42 --fps 8

# Watch at 30 FPS
python evaluate.py --games 100 --seed 42 --fps 30

# Watch at 60 FPS
python evaluate.py --games 100 --seed 42 --fps 60

# Evaluate 500 games
python evaluate.py --games 500 --seed 42

# Evaluate 1000 games without graphics
python evaluate.py --games 1000 --seed 42 --no-display
```

### MDP Output File

- `results/mdp_evaluation.csv` — evaluation results

---

## 2. Deep Q-Learning (DQN) Agent

### Change to DQN folder

```powershell
cd D:\projects\pythonProjects\PythonProject\dqagent
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Train DQN

```powershell
# Quick test: 10 episodes
python train.py --episodes 10

# Train for 100 episodes
python train.py --episodes 100

# Train for 2000 episodes
python train.py --episodes 2000

# Train for 5000 episodes
python train.py --episodes 5000

# Train using seed 42
python train.py --episodes 2000 --seed 42

# Train and save results to a different CSV
python train.py --episodes 2000 --log results/2000_episodes.csv
```

### Watch DQN play

```powershell
# Watch trained agent (8 FPS)
python main.py

# Watch untrained agent (8 FPS)
python main.py --untrained
```

### Evaluate DQN and record results

```powershell
# Watch 10 games (default: 30 FPS)
python evaluate.py --games 10

# Watch 100 games
python evaluate.py --games 100

# Watch 100 games using seed 42
python evaluate.py --games 100 --seed 42

# Watch at 8 FPS (same as main.py)
python evaluate.py --games 100 --seed 42 --fps 8

# Watch at 30 FPS
python evaluate.py --games 100 --seed 42 --fps 30

# Watch at 60 FPS
python evaluate.py --games 100 --seed 42 --fps 60

# Evaluate 500 games
python evaluate.py --games 500 --seed 42

# Evaluate 1000 games without graphics
python evaluate.py --games 1000 --seed 42 --no-display
```

### DQN Output Files

- `model/model.pth` — trained neural network
- `results/training_results.csv` — training results
- `results/dqn_evaluation.csv` — evaluation results

---

## 3. Recommended Comparison Workflow

### Step 1 — Evaluate MDP

```powershell
cd D:\projects\pythonProjects\PythonProject\MDP_Reward
python evaluate.py --games 100 --seed 42 --fps 30
```

### Step 2 — Train DQN

```powershell
cd D:\projects\pythonProjects\PythonProject\dqagent
python train.py --episodes 2000 --seed 42
```

### Step 3 — Evaluate DQN

```powershell
python evaluate.py --games 100 --seed 42 --fps 30
```

---