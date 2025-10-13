# Snake Game with Reinforcement Learning

A traditional Snake game implementation with reward system and logging capabilities, designed for reinforcement learning experiments.

## Features

- **Traditional Snake Game**: Classic snake gameplay with food collection and collision detection
- **Configurable Grid Size**: Default 15x20, user can specify custom dimensions
- **Reward System**: 
  - +1 for eating food
  - -1 for collision (wall or self)
  - 0 for empty moves
- **Game Logging**: Each game creates a timestamped log file in the `logs/` folder
- **Arrow Key Controls**: Use arrow keys to control the snake
- **RL Ready**: Includes skeleton for reinforcement learning algorithm implementation

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## How to Play

1. Run the game:
```bash
python snake_game.py
```

2. Enter grid size when prompted (or press Enter for default 15x20)

3. Use arrow keys to control the snake:
   - ↑ Up
   - ↓ Down  
   - ← Left
   - → Right

4. Game controls:
   - **R**: Restart game
   - **ESC**: Quit game

## Game Rules

- Snake moves continuously in the current direction
- Eating food increases score and snake length
- Colliding with walls or snake body ends the game
- Each move generates a reward: +1 (food), -1 (collision), 0 (empty)

## Logging

Each game session creates a detailed log file in the `logs/` folder with:
- Game timestamp and grid size
- Final score and total reward
- Complete move history with timestamps
- Direction, position, reward, and score for each move

## Reinforcement Learning

The `rl_algorithm.py` file contains a skeleton structure for implementing a Deep Q-Network (DQN) to train an AI agent to play Snake. The structure includes:

- `SnakeEnvironment`: RL-friendly game wrapper
- `DQN`: Neural network for Q-learning
- `ReplayBuffer`: Experience replay for training
- `DQNAgent`: Complete DQN agent implementation
- Training and testing functions

## File Structure

```
├── snake_game.py          # Main game implementation
├── rl_algorithm.py        # RL algorithm skeleton
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── logs/                 # Game log files (created automatically)
```

## Dependencies

- pygame >= 2.1.0
- numpy >= 1.21.0  
- torch >= 1.9.0

## Future Development

The RL algorithm skeleton is ready for implementation. Key areas to complete:

1. State representation for the neural network
2. Action space mapping
3. Training loop implementation
4. Model evaluation and testing
5. Integration with the main game

## Example Usage

```python
# Play the game manually
python snake_game.py

# The RL algorithm skeleton can be extended to:
# 1. Train an AI agent
# 2. Test trained models
# 3. Evaluate performance
```
