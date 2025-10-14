"""
Reinforcement Learning Algorithm for Snake Game

This file contains the skeleton structure for implementing a reinforcement learning
algorithm to train an AI agent to play the Snake game.

The algorithm will learn to:
- Navigate the snake without hitting walls or itself
- Collect food efficiently
- Maximize the total reward (+1 for food, -1 for collision, 0 for empty moves)
"""
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# Constants
STATE_SIZE = 26  # Size of state vector from get_state() function (23 + 3 additional features)

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
        
        # Game state variables
        self.snake = []
        self.food = (0, 0)
        self.direction = (1, 0)  # Moving right initially
        self.score = 0
        self.done = False
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self.action_map = {
            0: (0, -1),  # up
            1: (0, 1),   # down
            2: (-1, 0),  # left
            3: (1, 0)    # right
        }
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        # Reset game state
        self.done = False
        self.score = 0
        self.direction = (1, 0)  # Start moving right
        
        # Initialize snake in the center of the grid
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.snake = [
            (center_x, center_y),      # Head
            (center_x - 1, center_y),  # Body segment 1
            (center_x - 2, center_y)   # Body segment 2
        ]
        
        # Place food at a random empty position
        self._place_food()
        
        # Return initial state observation
        return self.get_state()
    
    def _place_food(self):
        """Place food at a random empty position."""
        while True:
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            # If game is already over, return current state with no reward
            return self.get_state(), 0.0, True, {"message": "Game already over"}
        
        # Store old head position for distance calculation
        old_head_x, old_head_y = self.snake[0]
        
        # Get the direction for the action
        new_direction = self.action_map[action]
        
        # Prevent snake from moving in opposite direction (into itself)
        if (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1]):
            # Invalid move - snake would move into itself
            new_direction = self.direction  # Keep current direction
        
        # Update direction
        self.direction = new_direction
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Initialize reward and info
        base_reward = 0.0
        info = {"action": action, "direction": self.direction}
        
        # Check for collisions
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or 
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            self.done = True
            base_reward = -1.0
            info["collision"] = "wall"
            return self.get_state(), base_reward, self.done, info
        
        # Self collision
        if new_head in self.snake:
            self.done = True
            base_reward = -1.0
            info["collision"] = "self"
            return self.get_state(), base_reward, self.done, info
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if food is eaten
        if new_head == self.food:
            # Food eaten - snake grows, get positive reward
            base_reward = 1.0
            self.score += 1
            info["food_eaten"] = True
            info["score"] = self.score
            
            # Place new food
            self._place_food()
        else:
            # No food eaten - remove tail
            self.snake.pop()
            base_reward = 0.0
            info["food_eaten"] = False
        
        # ENHANCED REWARD SHAPING: Add additional rewards for better learning
        total_reward = base_reward
        
        # Distance-based reward (encourage moving toward food)
        if not self.done:
            food_x, food_y = self.food
            old_distance = abs(food_x - old_head_x) + abs(food_y - old_head_y)
            new_distance = abs(food_x - new_head[0]) + abs(food_y - new_head[1])
            distance_reward = (old_distance - new_distance) * 0.2  # Increased from 0.1
            total_reward += distance_reward
            info["distance_reward"] = distance_reward
        
        # Survival reward (encourage staying alive)
        if not self.done:
            survival_reward = 0.02  # Increased from 0.01
            total_reward += survival_reward
            info["survival_reward"] = survival_reward
        
        # Efficiency reward (encourage shorter paths to food)
        if not self.done and base_reward == 0.0:  # Only for non-food moves
            food_x, food_y = self.food
            current_distance = abs(food_x - new_head[0]) + abs(food_y - new_head[1])
            max_distance = self.grid_width + self.grid_height
            efficiency_reward = (1.0 - current_distance / max_distance) * 0.01
            total_reward += efficiency_reward
            info["efficiency_reward"] = efficiency_reward
        
        # Length bonus (encourage growing the snake)
        if base_reward == 1.0:  # When food is eaten
            length_bonus = len(self.snake) * 0.1  # Bonus proportional to snake length
            total_reward += length_bonus
            info["length_bonus"] = length_bonus
        
        # Update info with current state
        info["snake_length"] = len(self.snake)
        info["head_position"] = new_head
        info["food_position"] = self.food
        info["base_reward"] = base_reward
        info["total_reward"] = total_reward
        
        # Get next state
        next_state = self.get_state()
        
        return next_state, total_reward, self.done, info
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state representation.
        
        Returns:
            State vector representing the current game state
        """
        # Get snake head position
        head_x, head_y = self.snake[0]
        
        # Get food position
        food_x, food_y = self.food
        
        # Calculate distances to walls
        wall_up = head_y
        wall_down = self.grid_height - 1 - head_y
        wall_left = head_x
        wall_right = self.grid_width - 1 - head_x
        
        # Calculate distance to food
        food_distance_x = food_x - head_x
        food_distance_y = food_y - head_y
        
        # Calculate Manhattan distance to food
        food_distance = abs(food_distance_x) + abs(food_distance_y)
        
        # Check for immediate dangers (collision in next move for each direction)
        danger_straight = 0
        danger_right = 0
        danger_left = 0
        
        # Current direction
        dx, dy = self.direction
        
        # Check straight ahead
        next_head = (head_x + dx, head_y + dy)
        if (next_head[0] < 0 or next_head[0] >= self.grid_width or 
            next_head[1] < 0 or next_head[1] >= self.grid_height or 
            next_head in self.snake):
            danger_straight = 1
        
        # Check right turn (relative to current direction)
        right_dx, right_dy = -dy, dx  # 90 degree right turn
        next_head_right = (head_x + right_dx, head_y + right_dy)
        if (next_head_right[0] < 0 or next_head_right[0] >= self.grid_width or 
            next_head_right[1] < 0 or next_head_right[1] >= self.grid_height or 
            next_head_right in self.snake):
            danger_right = 1
        
        # Check left turn (relative to current direction)
        left_dx, left_dy = dy, -dx  # 90 degree left turn
        next_head_left = (head_x + left_dx, head_y + left_dy)
        if (next_head_left[0] < 0 or next_head_left[0] >= self.grid_width or 
            next_head_left[1] < 0 or next_head_left[1] >= self.grid_height or 
            next_head_left in self.snake):
            danger_left = 1
        
        # Direction encoding (one-hot)
        direction_up = 1 if self.direction == (0, -1) else 0
        direction_down = 1 if self.direction == (0, 1) else 0
        direction_left = 1 if self.direction == (-1, 0) else 0
        direction_right = 1 if self.direction == (1, 0) else 0
        
        # Food direction relative to snake head
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        
        # Snake length (normalized)
        snake_length_norm = len(self.snake) / (self.grid_width * self.grid_height)
        
        # Additional features for better learning
        # 1. Food direction relative to head (more precise)
        food_dx = food_x - head_x
        food_dy = food_y - head_y
        
        # 2. Snake body density around head (danger awareness)
        body_density = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_pos = (head_x + dx, head_y + dy)
                if check_pos in self.snake:
                    body_density += 1
        
        # Create state vector
        state = np.array([
            # Snake head position (normalized)
            head_x / self.grid_width,
            head_y / self.grid_height,
            
            # Food position (normalized)
            food_x / self.grid_width,
            food_y / self.grid_height,
            
            # Distances to walls (normalized)
            wall_up / self.grid_height,
            wall_down / self.grid_height,
            wall_left / self.grid_width,
            wall_right / self.grid_width,
            
            # Distance to food
            food_distance_x / self.grid_width,
            food_distance_y / self.grid_height,
            food_distance / (self.grid_width + self.grid_height),
            
            # Danger detection
            danger_straight,
            danger_right,
            danger_left,
            
            # Direction encoding
            direction_up,
            direction_down,
            direction_left,
            direction_right,
            
            # Food direction
            food_up,
            food_down,
            food_left,
            food_right,
            
            # Snake length
            snake_length_norm,
            
            # Additional features
            food_dx / self.grid_width,  # Relative food position X
            food_dy / self.grid_height,  # Relative food position Y
            body_density / 8.0  # Body density around head (normalized)
        ], dtype=np.float32)
        
        return state
    
    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions (excluding opposite direction).
        
        Returns:
            List of valid action indices
        """
        valid_actions = []
        head_x, head_y = self.snake[0]
        
        # Check each possible action (0=up, 1=down, 2=left, 3=right)
        for action in range(4):
            direction = self.action_map[action]
            
            # Calculate next head position
            next_head = (head_x + direction[0], head_y + direction[1])
            
            # Check if this action would cause immediate collision
            # 1. Wall collision
            if (next_head[0] < 0 or next_head[0] >= self.grid_width or 
                next_head[1] < 0 or next_head[1] >= self.grid_height):
                continue  # Skip this action
            
            # 2. Self collision (check if next position is in snake body)
            if next_head in self.snake:
                continue  # Skip this action
            
            # 3. Opposite direction (prevent snake from going backwards into itself)
            # This is already handled in the step function, but we can be extra safe
            opposite_direction = (-self.direction[0], -self.direction[1])
            if direction == opposite_direction:
                continue  # Skip this action
            
            # If we get here, the action is valid
            valid_actions.append(action)
        
        # If no valid actions (shouldn't happen in normal gameplay), 
        # return all actions to let the game handle the collision
        if not valid_actions:
            return [0, 1, 2, 3]  # All actions
        
        return valid_actions

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
        
        # Define the neural network architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        # First layer: input -> hidden_size
        x = F.relu(self.fc1(x))
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.bn1(x)
        x = self.dropout1(x)
        
        # Second layer: hidden_size -> hidden_size
        x = F.relu(self.fc2(x))
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.bn2(x)
        x = self.dropout2(x)
        
        # Third layer: hidden_size -> hidden_size//2
        x = F.relu(self.fc3(x))
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.bn3(x)
        x = self.dropout3(x)
        
        # Output layer: hidden_size//2 -> output_size (no activation for Q-values)
        x = self.fc4(x)
        
        return x

class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """
    
    def __init__(self, capacity: int = 50000):
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
        # Convert inputs to appropriate types for storage
        if isinstance(state, np.ndarray):
            state = state.copy()  # Make a copy to avoid reference issues
        if isinstance(next_state, np.ndarray):
            next_state = next_state.copy()
        
        # Store the experience tuple
        experience = (state, action, float(reward), next_state, bool(done))
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        # Check if buffer has enough experiences
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Randomly sample experiences
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch into separate arrays
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent for Snake game.
    """
    
    def __init__(self, state_size: int, action_size: int = 4, lr: float = 0.0001):
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
        
        # ADVANCED Hyperparameters for better learning
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.02  # Lower minimum exploration for better exploitation
        self.epsilon_decay = 0.9995  # Even slower epsilon decay
        self.gamma = 0.99  # High discount factor for long-term planning
        self.batch_size = 128  # Larger batch size for more stable learning
        self.update_target_freq = 500  # More frequent target network updates
        
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
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random exploration
            return random.choice(range(self.action_size))
        else:
            # Exploitation: use neural network to select best action
            with torch.no_grad():  # No gradient computation needed for inference
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
                
                # Get Q-values from network
                q_values = self.q_network(state_tensor)
                
                # Select action with highest Q-value
                action = q_values.argmax().item()
                
                return action
    
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
        # Store the experience in the replay buffer
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """
        Train the agent on a batch of experiences.
        """
        # Check if we have enough experiences to train
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            # Get next state Q-values from target network
            next_q_values = self.target_network(next_states).max(1)[0]
            
            # Compute target Q-values using Bellman equation
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model checkpoint
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'update_target_freq': self.update_target_freq,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Load network states
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            
            # Load hyperparameters (if available)
            if 'state_size' in checkpoint:
                if checkpoint['state_size'] != self.state_size:
                    print(f"Warning: State size mismatch. Expected {self.state_size}, got {checkpoint['state_size']}")
            
            if 'action_size' in checkpoint:
                if checkpoint['action_size'] != self.action_size:
                    print(f"Warning: Action size mismatch. Expected {self.action_size}, got {checkpoint['action_size']}")
            
            # Load other hyperparameters if available
            if 'lr' in checkpoint:
                self.lr = checkpoint['lr']
            if 'gamma' in checkpoint:
                self.gamma = checkpoint['gamma']
            if 'batch_size' in checkpoint:
                self.batch_size = checkpoint['batch_size']
            if 'update_target_freq' in checkpoint:
                self.update_target_freq = checkpoint['update_target_freq']
            if 'epsilon_min' in checkpoint:
                self.epsilon_min = checkpoint['epsilon_min']
            if 'epsilon_decay' in checkpoint:
                self.epsilon_decay = checkpoint['epsilon_decay']
            
            print(f"Model loaded successfully from {filepath}")
            print(f"Epsilon: {self.epsilon:.4f}, Step count: {self.step_count}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model from {filepath}: {str(e)}")

def train_agent(episodes: int = 5000, grid_width: int = 15, grid_height: int = 20, curriculum: bool = True):
    """
    Train the DQN agent on the Snake game.
    
    Args:
        episodes: Number of training episodes
        grid_width: Width of the game grid
        grid_height: Height of the game grid
        curriculum: Whether to use curriculum learning (start with smaller grids)
    """
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Create environment and agent
    env = SnakeEnvironment(grid_width, grid_height)
    agent = DQNAgent(state_size=STATE_SIZE, action_size=4, lr=0.0001)
    
    # Training tracking
    scores = []
    avg_scores = []
    epsilons = []
    losses = []
    episode_rewards = []
    
    # Training parameters
    save_frequency = 100  # Save model every 100 episodes
    log_frequency = 10    # Log progress every 10 episodes
    
    # Curriculum learning parameters
    if curriculum:
        curriculum_stages = [
            (5, 5, episodes // 4),      # Stage 1: 5x5 grid for 25% of episodes
            (8, 8, episodes // 4),      # Stage 2: 8x8 grid for 25% of episodes
            (10, 10, episodes // 4),    # Stage 3: 10x10 grid for 25% of episodes
            (grid_width, grid_height, episodes // 4)  # Stage 4: Full grid for 25% of episodes
        ]
        current_stage = 0
        stage_episodes = 0
        current_width, current_height, stage_length = curriculum_stages[current_stage]
        print(f"Curriculum Learning: Starting with {current_width}x{current_height} grid")
    else:
        current_width, current_height = grid_width, grid_height
    
    print(f"Starting training for {episodes} episodes on {grid_width}x{grid_height} grid")
    print(f"Initial epsilon: {agent.epsilon:.4f}")
    
    # Training loop
    for episode in range(episodes):
        # Curriculum learning: switch to next stage if needed
        if curriculum and stage_episodes >= stage_length and current_stage < len(curriculum_stages) - 1:
            current_stage += 1
            stage_episodes = 0
            current_width, current_height, stage_length = curriculum_stages[current_stage]
            print(f"Curriculum Learning: Switching to {current_width}x{current_height} grid at episode {episode}")
            # Create new environment for new grid size
            env = SnakeEnvironment(current_width, current_height)
        
        state = env.reset()
        done = False
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        while not done:
            # Choose action
            action = agent.act(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_loss += loss
            
            # Update state and tracking
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Prevent infinite episodes
            if steps > 1000:
                done = True
        
        # Record episode results
        scores.append(env.score)
        episode_rewards.append(episode_reward)
        epsilons.append(agent.epsilon)
        if episode_loss > 0:
            losses.append(episode_loss / steps)
        
        # Calculate average score
        if len(scores) >= 100:
            avg_score = sum(scores[-100:]) / 100
        else:
            avg_score = sum(scores) / len(scores)
        avg_scores.append(avg_score)
        
        # Update curriculum learning counter
        if curriculum:
            stage_episodes += 1
        
        # Log progress
        if episode % log_frequency == 0:
            grid_info = f"({current_width}x{current_height})" if curriculum else ""
            print(f"Episode {episode:4d} | Score: {env.score:3d} | "
                  f"Avg Score: {avg_score:6.2f} | Epsilon: {agent.epsilon:.4f} | "
                  f"Steps: {steps:3d} | Reward: {episode_reward:6.2f} {grid_info}")
        
        # Save model periodically
        if episode % save_frequency == 0 and episode > 0:
            model_path = f"models/snake_dqn_episode_{episode}.pth"
            agent.save_model(model_path)
            print(f"Model saved at episode {episode}")
        
        # Early stopping if performance is good
        if len(avg_scores) >= 100 and avg_scores[-1] > 50:
            print(f"Early stopping: Average score {avg_scores[-1]:.2f} > 50")
            break
    
    # Save final model
    final_model_path = f"models/snake_dqn_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    agent.save_model(final_model_path)
    
    # Create visualizations
    create_training_plots(scores, avg_scores, epsilons, losses, episode_rewards)
    
    # Save training log
    save_training_log(scores, avg_scores, epsilons, losses, episode_rewards, 
                     grid_width, grid_height, episodes)
    
    print(f"\nTraining completed!")
    print(f"Final average score: {avg_scores[-1]:.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Final model saved to: {final_model_path}")

def create_training_plots(scores, avg_scores, epsilons, losses, episode_rewards):
    """Create and save training visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Snake DQN Training Progress', fontsize=16)
    
    # Plot 1: Scores
    axes[0, 0].plot(scores, alpha=0.3, color='blue', label='Episode Scores')
    axes[0, 0].plot(avg_scores, color='red', linewidth=2, label='Average Score (100 episodes)')
    axes[0, 0].set_title('Training Scores')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Epsilon decay
    axes[0, 1].plot(epsilons, color='green', linewidth=2)
    axes[0, 1].set_title('Epsilon Decay')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].grid(True)
    
    # Plot 3: Episode rewards
    axes[1, 0].plot(episode_rewards, alpha=0.7, color='purple')
    axes[1, 0].set_title('Episode Rewards')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Reward')
    axes[1, 0].grid(True)
    
    # Plot 4: Training loss
    if losses:
        axes[1, 1].plot(losses, color='orange', linewidth=2)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Loss')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No loss data available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Loss')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"visualizations/training_progress_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to: {plot_path}")

def save_training_log(scores, avg_scores, epsilons, losses, episode_rewards, 
                     grid_width, grid_height, episodes):
    """Save detailed training log."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f"logs/training_log_{timestamp}.txt"
    
    with open(log_path, 'w') as f:
        f.write(f"Snake DQN Training Log - {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Grid Size: {grid_width}x{grid_height}\n")
        f.write(f"Total Episodes: {episodes}\n")
        f.write(f"Final Average Score: {avg_scores[-1]:.2f}\n")
        f.write(f"Best Score: {max(scores)}\n")
        f.write(f"Final Epsilon: {epsilons[-1]:.4f}\n")
        f.write("\nEpisode Details:\n")
        f.write("Episode\tScore\tAvgScore\tEpsilon\tReward\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(scores)):
            f.write(f"{i}\t{scores[i]}\t{avg_scores[i]:.2f}\t{epsilons[i]:.4f}\t{episode_rewards[i]:.2f}\n")
    
    print(f"Training log saved to: {log_path}")

def test_agent(model_path: str, grid_width: int = 15, grid_height: int = 20, 
               num_episodes: int = 10, render: bool = True):
    """
    Test a trained agent on the Snake game.
    
    Args:
        model_path: Path to the trained model
        grid_width: Width of the game grid
        grid_height: Height of the game grid
        num_episodes: Number of test episodes to run
        render: Whether to render the game visually
    """
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("renderings", exist_ok=True)
    
    # Create environment and agent
    env = SnakeEnvironment(grid_width, grid_height)
    agent = DQNAgent(state_size=STATE_SIZE, action_size=4, lr=0.0001)
    
    # Load the trained model
    try:
        agent.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test tracking
    test_scores = []
    test_rewards = []
    test_steps = []
    episode_details = []
    
    print(f"Testing agent for {num_episodes} episodes on {grid_width}x{grid_height} grid")
    print("=" * 60)
    
    # Run test episodes
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_actions = []
        
        print(f"Episode {episode + 1}/{num_episodes}: ", end="", flush=True)
        
        while not done:
            # Choose action (no exploration during testing)
            action = agent.act(state, training=False)
            episode_actions.append(action)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update tracking
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Prevent infinite episodes
            if episode_steps > 1000:
                done = True
                print("(timeout)", end="")
        
        # Record episode results
        test_scores.append(env.score)
        test_rewards.append(episode_reward)
        test_steps.append(episode_steps)
        
        episode_details.append({
            'episode': episode + 1,
            'score': env.score,
            'reward': episode_reward,
            'steps': episode_steps,
            'actions': episode_actions.copy()
        })
        
        print(f"Score: {env.score}, Reward: {episode_reward:.2f}, Steps: {episode_steps}")
    
    # Calculate statistics
    avg_score = sum(test_scores) / len(test_scores)
    avg_reward = sum(test_rewards) / len(test_rewards)
    avg_steps = sum(test_steps) / len(test_steps)
    max_score = max(test_scores)
    min_score = min(test_scores)
    
    # Display results
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Best Score: {max_score}")
    print(f"Worst Score: {min_score}")
    print(f"Score Range: {max_score - min_score}")
    
    # Save results
    save_test_results(test_scores, test_rewards, test_steps, episode_details,
                     grid_width, grid_height, model_path, avg_score, max_score)
    
    # Create visualization
    create_test_plots(test_scores, test_rewards, test_steps, model_path)
    
    # Render best episode if requested
    if render and max_score > 0:
        render_best_episode(env, agent, episode_details, grid_width, grid_height)
    
    print(f"\nResults saved to results/ folder")
    if render:
        print(f"Renderings saved to renderings/ folder")

def save_test_results(test_scores, test_rewards, test_steps, episode_details,
                     grid_width, grid_height, model_path, avg_score, max_score):
    """Save detailed test results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f"results/test_results_{timestamp}.txt"
    
    with open(results_path, 'w') as f:
        f.write(f"Snake DQN Test Results - {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Grid Size: {grid_width}x{grid_height}\n")
        f.write(f"Episodes: {len(test_scores)}\n")
        f.write(f"Average Score: {avg_score:.2f}\n")
        f.write(f"Best Score: {max_score}\n")
        f.write(f"Average Reward: {sum(test_rewards)/len(test_rewards):.2f}\n")
        f.write(f"Average Steps: {sum(test_steps)/len(test_steps):.2f}\n")
        f.write("\nEpisode Details:\n")
        f.write("Episode\tScore\tReward\tSteps\tActions\n")
        f.write("-" * 50 + "\n")
        
        for detail in episode_details:
            actions_str = "".join([str(a) for a in detail['actions'][:20]])  # First 20 actions
            if len(detail['actions']) > 20:
                actions_str += "..."
            f.write(f"{detail['episode']}\t{detail['score']}\t{detail['reward']:.2f}\t"
                   f"{detail['steps']}\t{actions_str}\n")
    
    print(f"Test results saved to: {results_path}")

def create_test_plots(test_scores, test_rewards, test_steps, model_path):
    """Create and save test visualization plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Snake DQN Test Results', fontsize=16)
    
    # Plot 1: Scores
    axes[0].bar(range(1, len(test_scores) + 1), test_scores, color='blue', alpha=0.7)
    axes[0].axhline(y=sum(test_scores)/len(test_scores), color='red', linestyle='--', 
                   label=f'Average: {sum(test_scores)/len(test_scores):.1f}')
    axes[0].set_title('Test Scores')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rewards
    axes[1].bar(range(1, len(test_rewards) + 1), test_rewards, color='green', alpha=0.7)
    axes[1].axhline(y=sum(test_rewards)/len(test_rewards), color='red', linestyle='--',
                   label=f'Average: {sum(test_rewards)/len(test_rewards):.1f}')
    axes[1].set_title('Test Rewards')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Steps
    axes[2].bar(range(1, len(test_steps) + 1), test_steps, color='orange', alpha=0.7)
    axes[2].axhline(y=sum(test_steps)/len(test_steps), color='red', linestyle='--',
                   label=f'Average: {sum(test_steps)/len(test_steps):.1f}')
    axes[2].set_title('Test Steps')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Steps per Episode')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"results/test_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test plots saved to: {plot_path}")

def render_best_episode(env, agent, episode_details, grid_width, grid_height):
    """Render the best performing episode."""
    # Find best episode
    best_episode = max(episode_details, key=lambda x: x['score'])
    
    if best_episode['score'] == 0:
        print("No successful episodes to render")
        return
    
    print(f"\nRendering best episode (Score: {best_episode['score']})...")
    
    # Replay the best episode
    state = env.reset()
    done = False
    step = 0
    
    # Create rendering frames (simplified - in practice you'd use pygame or similar)
    rendering_data = []
    
    for action in best_episode['actions']:
        if done:
            break
            
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Record frame data
        frame_data = {
            'step': step,
            'action': action,
            'reward': reward,
            'score': env.score,
            'snake': env.snake.copy(),
            'food': env.food,
            'done': done
        }
        rendering_data.append(frame_data)
        
        state = next_state
        step += 1
    
    # Save rendering data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rendering_path = f"renderings/best_episode_{timestamp}.txt"
    
    with open(rendering_path, 'w') as f:
        f.write(f"Best Episode Rendering - Score: {best_episode['score']}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Grid Size: {grid_width}x{grid_height}\n")
        f.write(f"Total Steps: {len(rendering_data)}\n")
        f.write("\nStep-by-step details:\n")
        f.write("Step\tAction\tReward\tScore\tSnake Head\tFood\n")
        f.write("-" * 50 + "\n")
        
        for frame in rendering_data:
            snake_head = frame['snake'][0] if frame['snake'] else "N/A"
            f.write(f"{frame['step']}\t{frame['action']}\t{frame['reward']}\t"
                   f"{frame['score']}\t{snake_head}\t{frame['food']}\n")
    
    print(f"Best episode rendering saved to: {rendering_path}")
    print("Note: For visual rendering, integrate with pygame display from snake_game.py")

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
    print(f"Evaluating agent for {num_episodes} episodes...")
    
    # Performance tracking
    scores = []
    rewards = []
    steps = []
    survival_times = []
    food_eaten_count = []
    collision_types = {'wall': 0, 'self': 0}
    success_episodes = 0
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_food = 0
        
        while not done:
            # Choose action (no exploration during evaluation)
            action = agent.act(state, training=False)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Track metrics
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Count food eaten
            if reward == 1.0:
                episode_food += 1
            
            # Track collision type if game ended
            if done and episode_steps < 1000:  # Not a timeout
                if 'collision' in info:
                    collision_types[info['collision']] += 1
            
            # Prevent infinite episodes
            if episode_steps > 1000:
                done = True
        
        # Record episode results
        scores.append(env.score)
        rewards.append(episode_reward)
        steps.append(episode_steps)
        survival_times.append(episode_steps)
        food_eaten_count.append(episode_food)
        
        # Count successful episodes (score > 0)
        if env.score > 0:
            success_episodes += 1
        
        # Progress indicator
        if (episode + 1) % 20 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes...")
    
    # Calculate comprehensive metrics
    metrics = {
        # Basic performance metrics
        'avg_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'max_score': float(np.max(scores)),
        'min_score': float(np.min(scores)),
        'median_score': float(np.median(scores)),
        
        # Reward metrics
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'max_reward': float(np.max(rewards)),
        'min_reward': float(np.min(rewards)),
        
        # Survival metrics
        'avg_survival_time': float(np.mean(survival_times)),
        'std_survival_time': float(np.std(survival_times)),
        'max_survival_time': float(np.max(survival_times)),
        'min_survival_time': float(np.min(survival_times)),
        
        # Success metrics
        'success_rate': float(success_episodes / num_episodes),
        'avg_food_per_episode': float(np.mean(food_eaten_count)),
        'max_food_per_episode': float(np.max(food_eaten_count)),
        
        # Collision analysis
        'wall_collision_rate': float(collision_types['wall'] / num_episodes),
        'self_collision_rate': float(collision_types['self'] / num_episodes),
        'timeout_rate': float((num_episodes - collision_types['wall'] - collision_types['self']) / num_episodes),
        
        # Efficiency metrics
        'avg_steps_per_food': float(np.mean(steps) / np.mean(food_eaten_count)) if np.mean(food_eaten_count) > 0 else float('inf'),
        'food_efficiency': float(np.mean(food_eaten_count) / np.mean(steps)) if np.mean(steps) > 0 else 0.0,
        
        # Consistency metrics
        'score_consistency': float(1.0 - (np.std(scores) / np.mean(scores))) if np.mean(scores) > 0 else 0.0,
        'performance_stability': float(1.0 - (np.std(rewards) / np.mean(rewards))) if np.mean(rewards) > 0 else 0.0,
        
        # Additional statistics
        'total_episodes': num_episodes,
        'successful_episodes': success_episodes,
        'failed_episodes': num_episodes - success_episodes
    }
    
    # Display results
    print("\n" + "=" * 60)
    print("AGENT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes Evaluated: {num_episodes}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Average Score: {metrics['avg_score']:.2f} ± {metrics['std_score']:.2f}")
    print(f"Best Score: {metrics['max_score']:.0f}")
    print(f"Average Survival Time: {metrics['avg_survival_time']:.1f} steps")
    print(f"Average Food per Episode: {metrics['avg_food_per_episode']:.2f}")
    print(f"Food Efficiency: {metrics['food_efficiency']:.4f}")
    print(f"Wall Collision Rate: {metrics['wall_collision_rate']:.2%}")
    print(f"Self Collision Rate: {metrics['self_collision_rate']:.2%}")
    print(f"Timeout Rate: {metrics['timeout_rate']:.2%}")
    print(f"Score Consistency: {metrics['score_consistency']:.3f}")
    
    return metrics

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
