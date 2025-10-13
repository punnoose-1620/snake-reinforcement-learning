"""
Reinforcement Learning Algorithm for Snake Game

This file contains the skeleton structure for implementing a reinforcement learning
algorithm to train an AI agent to play the Snake game.

The algorithm will learn to:
- Navigate the snake without hitting walls or itself
- Collect food efficiently
- Maximize the total reward (+1 for food, -1 for collision, 0 for empty moves)
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SnakeEnvironment:
    """
    Environment wrapper for the Snake game that provides RL-friendly interface.
    """
    
    def __init__(self, grid_width: int = 15, grid_height: int = 20):
        """
        Initialize the Snake environment.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        # TODO: Implement environment reset
        # This should reset the snake game and return the initial state
        pass
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # TODO: Implement action execution
        # This should take an action, update the game state, and return:
        # - next_state: The new state after taking the action
        # - reward: The reward for this action (+1, -1, or 0)
        # - done: Whether the game is over
        # - info: Additional information (optional)
        pass
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state representation.
        
        Returns:
            State vector representing the current game state
        """
        # TODO: Implement state representation
        # This should return a numerical representation of the current state
        # Consider including:
        # - Snake position and direction
        # - Food position
        # - Distance to walls
        # - Distance to food
        # - Snake body positions
        pass
    
    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions (excluding opposite direction).
        
        Returns:
            List of valid action indices
        """
        # TODO: Implement valid action detection
        # This should return actions that don't cause immediate collision
        pass

class DQN(nn.Module):
    """
    Deep Q-Network for Snake game.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 4):
        """
        Initialize the DQN.
        
        Args:
            input_size: Size of the input state vector
            hidden_size: Size of hidden layers
            output_size: Number of possible actions (4 for Snake)
        """
        super(DQN, self).__init__()
        
        # TODO: Define the neural network architecture
        # Consider using:
        # - Fully connected layers
        # - Batch normalization
        # - Dropout for regularization
        # - ReLU activation functions
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        # TODO: Implement forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # TODO: Implement experience storage
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Batch of experiences
        """
        # TODO: Implement random sampling
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent for Snake game.
    """
    
    def __init__(self, state_size: int, action_size: int = 4, lr: float = 0.001):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of the state space
            action_size: Number of possible actions
            lr: Learning rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        # Neural networks
        self.q_network = DQN(state_size, output_size=action_size)
        self.target_network = DQN(state_size, output_size=action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        self.update_target_freq = 1000
        
        # Replay buffer
        self.memory = ReplayBuffer()
        
        # Training tracking
        self.step_count = 0
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action to take
        """
        # TODO: Implement epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            # TODO: Use neural network to select best action
            pass
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # TODO: Implement experience storage
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """
        Train the agent on a batch of experiences.
        """
        # TODO: Implement DQN training
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)
        
        # TODO: Convert batch to tensors
        # TODO: Compute current Q-values
        # TODO: Compute target Q-values
        # TODO: Compute loss and update network
        # TODO: Update target network periodically
        # TODO: Decay epsilon
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        # TODO: Implement model saving
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        # TODO: Implement model loading
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']

def train_agent(episodes: int = 1000, grid_width: int = 15, grid_height: int = 20):
    """
    Train the DQN agent on the Snake game.
    
    Args:
        episodes: Number of training episodes
        grid_width: Width of the game grid
        grid_height: Height of the game grid
    """
    # TODO: Implement training loop
    # This should:
    # 1. Create environment and agent
    # 2. Run episodes
    # 3. Collect experiences
    # 4. Train the agent
    # 5. Log training progress
    # 6. Save the trained model
    
    print("Training function not yet implemented.")
    print("This is a skeleton for future RL implementation.")

def test_agent(model_path: str, grid_width: int = 15, grid_height: int = 20):
    """
    Test a trained agent on the Snake game.
    
    Args:
        model_path: Path to the trained model
        grid_width: Width of the game grid
        grid_height: Height of the game grid
    """
    # TODO: Implement agent testing
    # This should:
    # 1. Load the trained model
    # 2. Run the agent on the game
    # 3. Display the results
    # 4. Optionally render the game
    
    print("Testing function not yet implemented.")
    print("This is a skeleton for future RL implementation.")

def evaluate_agent(agent, env, num_episodes: int = 100) -> Dict[str, float]:
    """
    Evaluate the performance of a trained agent.
    
    Args:
        agent: Trained DQN agent
        env: Snake environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    # TODO: Implement agent evaluation
    # This should:
    # 1. Run the agent for multiple episodes
    # 2. Collect performance metrics
    # 3. Return average scores, success rate, etc.
    
    print("Evaluation function not yet implemented.")
    return {}

if __name__ == "__main__":
    print("RL Algorithm Skeleton for Snake Game")
    print("=" * 40)
    print("This file contains the structure for implementing")
    print("a reinforcement learning algorithm to train an AI")
    print("agent to play the Snake game.")
    print("\nTo implement the algorithm:")
    print("1. Complete the SnakeEnvironment class")
    print("2. Implement the DQN neural network")
    print("3. Add training and testing functions")
    print("4. Integrate with the main Snake game")
    print("\nRun 'python snake_game.py' to play the game manually.")
