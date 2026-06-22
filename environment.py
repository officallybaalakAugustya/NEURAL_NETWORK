import pygame
import random
import math

class SurvivalEnv:
    def __init__(self, render_mode=False):
        self.width = 800
        self.height = 600
        self.step_size = 20
        self.render_mode = render_mode
        self.current_steps = 0
        
        # YOUR IDEA: Multiple items
        self.num_foods = 5
        self.num_poisons = 3
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Q-Learning Survival Agent")
            self.clock = pygame.time.Clock()

        self.agent_pos = [self.width // 2, self.height // 2]
        self.foods = [self._random_pos() for _ in range(self.num_foods)]
        self.poisons = [self._random_pos() for _ in range(self.num_poisons)]

    def _random_pos(self):
        x = random.randint(1, (self.width // self.step_size) - 2) * self.step_size
        y = random.randint(1, (self.height // self.step_size) - 2) * self.step_size
        return [x, y]

    def _get_closest(self, target_list):
        closest = target_list[0]
        min_dist = float('inf')
        for item in target_list:
            dist = abs(item[0] - self.agent_pos[0]) + abs(item[1] - self.agent_pos[1])
            if dist < min_dist:
                min_dist = dist
                closest = item
        return closest

    def get_state(self):
        # AI SENSORS: Track the closest food and poison
        closest_food = self._get_closest(self.foods)
        closest_poison = self._get_closest(self.poisons)

        dx_food = (closest_food[0] - self.agent_pos[0]) / self.width
        dy_food = (closest_food[1] - self.agent_pos[1]) / self.height
        
        dx_poison = (closest_poison[0] - self.agent_pos[0]) / self.width
        dy_poison = (closest_poison[1] - self.agent_pos[1]) / self.height
        
        dist_top = self.agent_pos[1] / self.height
        dist_bottom = (self.height - self.agent_pos[1]) / self.height
        dist_left = self.agent_pos[0] / self.width
        dist_right = (self.width - self.agent_pos[0]) / self.width
        
        return [dx_food, dy_food, dx_poison, dy_poison, dist_top, dist_bottom, dist_left, dist_right]

    def step(self, action):
        old_x = self.agent_pos[0]
        old_y = self.agent_pos[1]

        if action == 0: self.agent_pos[1] -= self.step_size 
        if action == 1: self.agent_pos[1] += self.step_size 
        if action == 2: self.agent_pos[0] -= self.step_size 
        if action == 3: self.agent_pos[0] += self.step_size 

        self.agent_pos[0] = max(0, min(self.width - self.step_size, self.agent_pos[0]))
        self.agent_pos[1] = max(0, min(self.height - self.step_size, self.agent_pos[1]))
        
        self.current_steps += 1 
        
        # The Urgency Penalty
        reward = -0.1
        done = False

        # Hit Wall
        if self.agent_pos[0] == old_x and self.agent_pos[1] == old_y:
            reward = -10.0  
            done = True     

        # Eat Food (Checks if agent is on ANY of the foods)
        elif self.agent_pos in self.foods:
            reward = 10.0
            # Replace the eaten food with a new one
            food_index = self.foods.index(self.agent_pos)
            self.foods[food_index] = self._random_pos() 
            
        # Hit Poison (Checks if agent is on ANY of the poisons)
        elif self.agent_pos in self.poisons:
            reward = -10.0                 
            done = True                  

        if self.current_steps >= 500:
            done = True  

        return self.get_state(), reward, done

    def render(self):
        if not self.render_mode:
            return
        
        self.screen.fill((0, 0, 0)) 
        
        # Draw Agent
        pygame.draw.rect(self.screen, (0, 0, 255), (*self.agent_pos, self.step_size, self.step_size)) 
        
        # Draw All Foods
        for food in self.foods:
            pygame.draw.rect(self.screen, (0, 255, 0), (*food, self.step_size, self.step_size))  
            
        # Draw All Poisons
        for poison in self.poisons:
            pygame.draw.rect(self.screen, (255, 0, 0), (*poison, self.step_size, self.step_size)) 
        
        pygame.display.flip() 
        self.clock.tick(30)   
    
    def reset(self):
        self.agent_pos = [self.width // 2, self.height // 2]
        self.foods = [self._random_pos() for _ in range(self.num_foods)]
        self.poisons = [self._random_pos() for _ in range(self.num_poisons)]
        self.current_steps = 0  
        return self.get_state()