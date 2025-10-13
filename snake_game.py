import pygame
import random
import time
import os
from datetime import datetime
from typing import Tuple, List, Optional

class SnakeGame:
    def __init__(self, grid_width: int = 15, grid_height: int = 20):
        """Initialize the Snake game with specified grid dimensions."""
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = 30  # Size of each cell in pixels
        
        # Calculate window dimensions
        self.window_width = self.grid_width * self.cell_size
        self.window_height = self.grid_height * self.cell_size
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Snake Game - Reinforcement Learning")
        self.clock = pygame.time.Clock()
        
        # Game state
        self.reset_game()
        
        # Create log folder
        self.log_folder = "logs"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        
        # Initialize logging
        self.start_time = datetime.now()
        self.log_file = os.path.join(self.log_folder, f"game_{self.start_time.strftime('%Y%m%d_%H%M%S')}.txt")
        self.log_moves = []
        
    def reset_game(self):
        """Reset the game to initial state."""
        # Snake starts in the middle, moving right
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.snake = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        self.direction = (1, 0)  # Moving right
        self.next_direction = (1, 0)
        
        # Place food randomly
        self.place_food()
        
        # Game state
        self.game_over = False
        self.score = 0
        self.total_reward = 0
        
    def place_food(self):
        """Place food at a random empty position."""
        while True:
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves (excluding opposite direction)."""
        current_direction = self.direction
        valid_moves = []
        
        # All possible directions
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        
        for direction in directions:
            # Don't allow moving in opposite direction
            if direction != (-current_direction[0], -current_direction[1]):
                valid_moves.append(direction)
        
        return valid_moves
    
    def move_snake(self, direction: Optional[Tuple[int, int]] = None) -> int:
        """Move the snake and return the reward for this move."""
        if direction is not None:
            self.next_direction = direction
        
        # Update direction
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Check for wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or 
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            self.game_over = True
            reward = -1
        # Check for self collision
        elif new_head in self.snake:
            self.game_over = True
            reward = -1
        # Check for food
        elif new_head == self.food:
            self.snake.insert(0, new_head)
            self.score += 1
            self.place_food()
            reward = 1
        # Normal move
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = 0
        
        self.total_reward += reward
        
        # Log the move
        move_log = {
            'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'direction': self.direction,
            'head_position': new_head,
            'reward': reward,
            'score': self.score,
            'snake_length': len(self.snake)
        }
        self.log_moves.append(move_log)
        
        return reward
    
    def handle_input(self):
        """Handle keyboard input for snake control."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != (0, 1):
                    self.next_direction = (0, -1)
                elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                    self.next_direction = (0, 1)
                elif event.key == pygame.K_LEFT and self.direction != (1, 0):
                    self.next_direction = (-1, 0)
                elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                    self.next_direction = (1, 0)
                elif event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def draw(self):
        """Draw the game on the screen."""
        self.screen.fill(self.BLACK)
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            x = segment[0] * self.cell_size
            y = segment[1] * self.cell_size
            color = self.GREEN if i == 0 else self.BLUE  # Head is brighter green
            pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, self.WHITE, (x, y, self.cell_size, self.cell_size), 1)
        
        # Draw food
        food_x = self.food[0] * self.cell_size
        food_y = self.food[1] * self.cell_size
        pygame.draw.rect(self.screen, self.RED, (food_x, food_y, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, self.WHITE, (food_x, food_y, self.cell_size, self.cell_size), 1)
        
        # Draw grid lines
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.window_width, y))
        
        # Draw score and reward info
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        reward_text = font.render(f"Total Reward: {self.total_reward}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(reward_text, (10, 50))
        
        # Draw game over message
        if self.game_over:
            game_over_font = pygame.font.Font(None, 48)
            game_over_text = game_over_font.render("GAME OVER", True, self.RED)
            restart_text = font.render("Press R to restart or ESC to quit", True, self.WHITE)
            
            text_rect = game_over_text.get_rect(center=(self.window_width//2, self.window_height//2 - 20))
            restart_rect = restart_text.get_rect(center=(self.window_width//2, self.window_height//2 + 20))
            
            self.screen.blit(game_over_text, text_rect)
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def save_log(self):
        """Save the game log to a file."""
        with open(self.log_file, 'w') as f:
            f.write(f"Snake Game Log - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Grid Size: {self.grid_width}x{self.grid_height}\n")
            f.write(f"Final Score: {self.score}\n")
            f.write(f"Total Reward: {self.total_reward}\n")
            f.write(f"Total Moves: {len(self.log_moves)}\n")
            f.write("-" * 50 + "\n")
            f.write("Move Log:\n")
            f.write("Time\t\tDirection\tHead Pos\tReward\tScore\tLength\n")
            f.write("-" * 50 + "\n")
            
            for move in self.log_moves:
                f.write(f"{move['timestamp']}\t{move['direction']}\t\t{move['head_position']}\t\t{move['reward']}\t\t{move['score']}\t\t{move['snake_length']}\n")
    
    def run(self):
        """Main game loop."""
        running = True
        move_timer = 0
        move_delay = 150  # milliseconds between moves
        
        print("Snake Game Controls:")
        print("- Arrow keys: Move snake")
        print("- R: Restart game")
        print("- ESC: Quit game")
        print(f"Grid size: {self.grid_width}x{self.grid_height}")
        
        while running:
            dt = self.clock.tick(60)  # 60 FPS
            move_timer += dt
            
            # Handle input
            running = self.handle_input()
            
            # Move snake at regular intervals
            if move_timer >= move_delay and not self.game_over:
                self.move_snake()
                move_timer = 0
            
            # Draw everything
            self.draw()
        
        # Save log when game ends
        self.save_log()
        print(f"Game log saved to: {self.log_file}")
        
        pygame.quit()

def get_grid_size():
    """Get grid size from user input."""
    print("Enter grid size (width x height) or press Enter for default (15x20):")
    user_input = input().strip()
    
    if not user_input:
        return 15, 20
    
    try:
        if 'x' in user_input.lower():
            width, height = user_input.lower().split('x')
            width, height = int(width.strip()), int(height.strip())
        else:
            # Assume square grid
            size = int(user_input)
            width, height = size, size
        
        # Validate grid size
        if width < 5 or height < 5:
            print("Grid size too small. Using minimum size 5x5.")
            return 5, 5
        elif width > 50 or height > 50:
            print("Grid size too large. Using maximum size 50x50.")
            return 50, 50
        
        return width, height
    except ValueError:
        print("Invalid input. Using default size 15x20.")
        return 15, 20

if __name__ == "__main__":
    # Get grid size from user
    grid_width, grid_height = get_grid_size()
    
    # Create and run the game
    game = SnakeGame(grid_width, grid_height)
    game.run()
