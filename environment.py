import pygame
import random

class SurvivalEnv:
    def __init__(self, render_mode=False):
        # 1. Initialize the Pygame Engine
        self.width = 800
        self.height = 600
        self.step_size = 20
        self.render_mode = render_mode # Save the toggle state
        self.current_steps = 0 # Safety initialization
        
        # ONLY boot up the heavy Pygame engine if we want to watch
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Q-Learning Survival Agent")
            self.clock = pygame.time.Clock()

        # 2. Define the World Grid
        self.step_size = 20 # The agent moves 20 pixels at a time
        
        # 3. Entity Starting Positions
        self.agent_pos = [self.width // 2, self.height // 2] # Starts in the exact center
        self.food_pos = self._random_pos()
        self.poison_pos = self._random_pos()

    def _random_pos(self):
        """Helper function to snap entities to the 20-pixel grid"""
        x = random.randint(1, (self.width // self.step_size) - 2) * self.step_size
        y = random.randint(1, (self.height // self.step_size) - 2) * self.step_size
        return [x, y]

    def get_state(self):
        """
        The "Sensors". Calculates the distance to objects AND WALLS.
        Normalizes the values for the Neural Network.
        """
        # Delta X and Y for Food
        dx_food = (self.food_pos[0] - self.agent_pos[0]) / self.width
        dy_food = (self.food_pos[1] - self.agent_pos[1]) / self.height
        
        # Delta X and Y for Poison
        dx_poison = (self.poison_pos[0] - self.agent_pos[0]) / self.width
        dy_poison = (self.poison_pos[1] - self.agent_pos[1]) / self.height
        
        # NEW: Distance to Walls (Normalized between 0.0 and 1.0)
        dist_top = self.agent_pos[1] / self.height
        dist_bottom = (self.height - self.agent_pos[1]) / self.height
        dist_left = self.agent_pos[0] / self.width
        dist_right = (self.width - self.agent_pos[0]) / self.width
        
        # The AI now has 8 inputs!
        return [dx_food, dy_food, dx_poison, dy_poison, dist_top, dist_bottom, dist_left, dist_right]

    def step(self, action):
        """
        Executes the AI's chosen action and calculates the consequences.
        """
        # --- THE FIX 1: Remember where we were before moving ---
        old_x = self.agent_pos[0]
        old_y = self.agent_pos[1]

        # 1. Move the Agent
        if action == 0: self.agent_pos[1] -= self.step_size # UP
        if action == 1: self.agent_pos[1] += self.step_size # DOWN
        if action == 2: self.agent_pos[0] -= self.step_size # LEFT
        if action == 3: self.agent_pos[0] += self.step_size # RIGHT

        # 2. Enforce Wall Boundaries
        self.agent_pos[0] = max(0, min(self.width - self.step_size, self.agent_pos[0]))
        self.agent_pos[1] = max(0, min(self.height - self.step_size, self.agent_pos[1]))
        
        self.current_steps += 1 
        
        # 3. Initialize default feedback
        reward = -0.1
        done = False

        # --- THE FIX 2: Penalize Wall Hits! ---
        if self.agent_pos[0] == old_x and self.agent_pos[1] == old_y:
            # The agent tried to move, but the wall blocked it.
            reward = -5.0  # OUCH! Stop hugging the wall!

        # 4. Collision Detection 
        if self.agent_pos == self.food_pos:
            reward = 10                  
            self.food_pos = self._random_pos() 
            
        elif self.agent_pos == self.poison_pos:
            reward = -10                 
            done = True                  

        if self.current_steps >= 500:
            done = True  
        
        return self.get_state(), reward, done
    
    def render(self):
        """Draws the current state of the world to the screen."""
        if not self.render_mode:
            return
        
        self.screen.fill((0, 0, 0)) # Wipe the screen black
        
        # Draw Entities: (Surface, Color(RGB), (X, Y, Width, Height))
        pygame.draw.rect(self.screen, (0, 0, 255), (*self.agent_pos, self.step_size, self.step_size)) # Blue Agent
        pygame.draw.rect(self.screen, (0, 255, 0), (*self.food_pos, self.step_size, self.step_size))  # Green Food
        pygame.draw.rect(self.screen, (255, 0, 0), (*self.poison_pos, self.step_size, self.step_size)) # Red Poison
        
        pygame.display.flip() # Push the drawing to the monitor
        self.clock.tick(30)   # Limit to 30 Frames Per Second so we can actually see it
    
    def reset(self):
        self.agent_pos = [self.width // 2, self.height // 2]
        self.food_pos = self._random_pos()
        self.poison_pos = self._random_pos()
        
        self.current_steps = 0  # Reset the clock!

        # 1. spawn food anywhere EXCEPT where the agent is
        while self.food_pos == self.agent_pos:
            self.food_pos = self._random_pos()
            
        # 2. Spawn poison anywhere EXCEPT where the agent or food is
        while self.poison_pos == self.agent_pos or self.poison_pos == self.food_pos:
            self.poison_pos = self._random_pos()
        
        return self.get_state()
