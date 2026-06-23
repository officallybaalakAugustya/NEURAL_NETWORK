Q-Learning Survival Agent
A custom Reinforcement Learning environment and Deep Q-Network (DQN) built entirely from scratch using Python and NumPy. The agent learns to navigate a dynamic 2D environment, hunt food, and avoid obstacles using purely mathematical optimization—no external machine learning libraries (like PyTorch or TensorFlow) were used.

Architecture Overview
We separated the system into three core components to keep the data flow clean:
The Sandbox (environment.py): A custom 2D grid built with Pygame. It handles collision physics, dynamic entity spawning (multiple food/poison nodes), and issues dense, distance-based rewards to solve the sparse-reward problem.
The Brain (model.py): A deep neural network built in pure NumPy. It features dual hidden layers (16 neurons each), Leaky ReLU activations to prevent dead neurons during negative reward shocks, and a custom-built Adam Optimizer to prevent Q-value oscillation.
The Pipeline (train.py): The central training loop. It utilizes an Experience Replay memory buffer to sample historical states, calculates the Mean Squared Error (MSE) of the Bellman equation, and triggers backpropagation.
Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
pip install -r requirements.txt


How to Run
1. Watch the Trained Agent (Testing)
We have included the finalized brain_weights.npy file in the repository. To watch the fully trained agent navigate the environment in real-time:
python test.py


2. Train a New Agent
To train the agent from scratch, run the training pipeline. This runs headlessly (without rendering the Pygame UI) for maximum speed.
python train.py


Note: Training will output a new brain_weights.npy file, as well as reward_history.csv and loss_history.csv for metric tracking.
Data & Visualization
The training metrics are saved to CSV formats. You can use Matplotlib alongside the generated CSV files to graph the agent's learning curve and loss reduction over the 800 episodes.
