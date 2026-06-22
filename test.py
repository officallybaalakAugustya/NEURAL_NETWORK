import time
import numpy as np
import pygame  
from environment import SurvivalEnv  # Update this if Shivansh named it differently
from model import QNetwork

def main():
    print("Loading Environment and AI Brain...")
    
    # ==========================================
    # THE FIX: Start the Pygame Engine First!
    # ==========================================
    pygame.init() 
    
    env = SurvivalEnv(render_mode=True)  # Ensure the environment is in render mode
    model = QNetwork(input_size=8, hidden_size=8, output_size=4)
    model.load_weights("brain_weights.npy")
    
    state = env.reset() if hasattr(env, 'reset') else env._get_state()
    done = False
    
    print("Starting Gameplay... Watch the Pygame window!")
    
    while not done:
        # Pygame Heartbeat
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # The AI predicts the best move
        q_values = model.forward(state)
        action = np.argmax(q_values)
        
        # The environment takes the step
        step_result = env.step(action)
        state = step_result[0]
        reward = step_result[1]
        done = step_result[2]
        
        # Force the Screen to Draw
        if hasattr(env, 'render'):
            env.render()
        
        # Slow down the frames 
        time.sleep(0.1)
        
    print(f"Game Over! Final tick reward: {reward}")
    pygame.quit()

if __name__ == "__main__":
    main()