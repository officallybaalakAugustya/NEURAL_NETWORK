import pygame
import random

class SurvivalEnv:
    def __init__(self, render_mode=False):
        self.width = 800
        self.height = 600
        self.step_size = 20
        self.render_mode = render_mode
        self.current_steps = 0
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Q-Learning Survival Agent")
            self.clock = pygame.time.Clock()

        self.agent_pos = [self.width // 2, self.height // 2]
        self.food_pos = self._random_pos()
        self.poison_pos = self._random_pos()

    def _random_pos(self):
        x = random.randint(1, (self.width // self.step_size) - 2) * self.step_size
        y = random.randint(1, (self.height // self.step_size) - 2) * self.step_size
        return [x, y]

    def get_state(self):
        dx_food = (self.food_pos[0] - self.agent_pos[0]) / self.width
        dy_food = (self.food_pos[1] - self.agent_pos[1]) / self.height
        
        dx_poison = (self.poison_pos[0] - self.agent_pos[0]) / self.width
        dy_poison = (self.poison_pos[1] - self.agent_pos[1]) / self.height
        
        dist_top = self.agent_pos[1] / self.height
        dist_bottom = (self.height - self.agent_pos[1]) / self.height
        dist_left = self.agent_pos[0] / self.width
        dist_right = (self.width - self.agent_pos[0]) / self.width
        
        return [dx_food, dy_food, dx_poison, dy_poison, dist_top, dist_bottom, dist_left, dist_right]

    def step(self, action):
        old_x = self.agent_pos[0]
        old_y = self.agent_pos[1]

        # Move the Agent
        if action == 0: self.agent_pos[1] -= self.step_size 
        if action == 1: self.agent_pos[1] += self.step_size 
        if action == 2: self.agent_pos[0] -= self.step_size 
        if action == 3: self.agent_pos[0] += self.step_size 

        # Enforce Boundaries
        self.agent_pos[0] = max(0, min(self.width - self.step_size, self.agent_pos[0]))
        self.agent_pos[1] = max(0, min(self.height - self.step_size, self.agent_pos[1]))
        
        self.current_steps += 1 
        
        # ==========================================
        # YOUR BRILLIANT FIX: Movement is free!
        # ==========================================
        reward = 0.0
        done = False

        # Collision Detection: Walls (Instant Death)
        if self.agent_pos[0] == old_x and self.agent_pos[1] == old_y:
            reward = -10.0  
            done = True     

        # Collision Detection: Food & Poison
        elif self.agent_pos == self.food_pos:
            reward = 10.0                  
            self.food_pos = self._random_pos() 
            
        elif self.agent_pos == self.poison_pos:
            reward = -10.0                 
            done = True                  

        if self.current_steps >= 500:
            done = True  

        return self.get_state(), reward, done

    def render(self):
        if not self.render_mode:
            return
        
        self.screen.fill((0, 0, 0)) 
        
        pygame.draw.rect(self.screen, (0, 0, 255), (*self.agent_pos, self.step_size, self.step_size)) 
        pygame.draw.rect(self.screen, (0, 255, 0), (*self.food_pos, self.step_size, self.step_size))  
        pygame.draw.rect(self.screen, (255, 0, 0), (*self.poison_pos, self.step_size, self.step_size)) 
        
        pygame.display.flip() 
        self.clock.tick(30)   
    
    def reset(self):
        self.agent_pos = [self.width // 2, self.height // 2]
        self.food_pos = self._random_pos()
        self.poison_pos = self._random_pos()
        
        self.current_steps = 0  

        while self.food_pos == self.agent_pos:
            self.food_pos = self._random_pos()
            
        while self.poison_pos == self.agent_pos or self.poison_pos == self.food_pos:
            self.poison_pos = self._random_pos()
        
        return self.get_state()