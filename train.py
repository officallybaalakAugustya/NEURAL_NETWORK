"""
train.py — Training Controller for Deep Q-Network (DQN) Reinforcement Learning

This module orchestrates the training loop for a reinforcement learning agent.
It connects the SurvivalEnv (environment.py) and the QNetwork (model.py),
implementing experience replay, epsilon-greedy exploration, and metrics logging.

Compatibility notes (relative to original spec):
    - Environment class is SurvivalEnv, not Environment
    - Model class is QNetwork, not NeuralNetwork
    - env.step(action) returns (next_state, reward, done)  [not step_action]
    - model.forward(state) returns Q-values               [not forward_state]
    - model.backward(s, a, r, ns, done) trains per sample [not backward_batch]
    - model.save_weights() is absent; gracefully handled

Usage:
    python train.py
"""

import random
import logging
from collections import deque

import numpy as np
import pandas as pd

from environment import SurvivalEnv   # teammate class name
from model import QNetwork             # teammate class name

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPISODES: int = 1000          # Total number of training episodes
BATCH_SIZE: int = 32          # Mini-batch size for experience replay
MEMORY_SIZE: int = 10_000     # Maximum capacity of the replay buffer

GAMMA: float = 0.99           # Discount factor for future rewards

EPSILON: float = 1.0          # Initial exploration probability
EPSILON_MIN: float = 0.05     # Minimum exploration probability
EPSILON_DECAY: float = 0.995  # Multiplicative decay applied per episode

NUM_ACTIONS: int = 4          # Number of discrete actions [0, 1, 2, 3]


class ReplayMemory:
    """
    Fixed-size circular buffer that stores transition experiences for
    off-policy learning via experience replay.

    Each experience is a tuple of:
        (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int = MEMORY_SIZE) -> None:
        """
        Initialise the replay buffer.

        Args:
            capacity: Maximum number of experiences to retain.
        """
        self.buffer: deque = deque(maxlen=capacity)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Append a single transition to the buffer.

        Args:
            state:      Current observation vector (shape (4,)).
            action:     Action index taken in *state*.
            reward:     Scalar reward received after taking *action*.
            next_state: Observation vector after the transition (shape (4,)).
            done:       Whether the episode terminated after this step.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = BATCH_SIZE) -> list:
        """
        Uniformly sample a mini-batch from the buffer.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A list of *batch_size* experience tuples.

        Raises:
            ValueError: If the buffer contains fewer experiences than
                        the requested batch size.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} experiences from a buffer "
                f"of size {len(self.buffer)}."
            )
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return the current number of stored experiences."""
        return len(self.buffer)


class Trainer:
    """
    Manages the full training lifecycle: environment interaction, action
    selection, experience storage, network updates, and metrics logging.
    """

    def __init__(self) -> None:
        """Initialise the environment, model, replay memory, and metrics."""
        # Core components
        self.env = SurvivalEnv()           # teammate class: SurvivalEnv
        self.model = QNetwork()            # teammate class: QNetwork
        self.memory = ReplayMemory(capacity=MEMORY_SIZE)

        # Exploration schedule
        self.epsilon: float = EPSILON

        # Metrics accumulators
        self.reward_history: list[float] = []
        self.loss_history: list[float] = []

    # -----------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        """
        Choose an action using an epsilon-greedy policy.

        With probability *epsilon* a random action is selected (exploration);
        otherwise the action with the highest predicted Q-value is chosen
        (exploitation).

        Args:
            state: Current observation vector (shape (4,)).

        Returns:
            An integer action index in [0, 1, 2, 3].
        """
        if random.random() < self.epsilon:
            # Exploration: uniformly random action
            return random.randint(0, NUM_ACTIONS - 1)

        # Exploitation: greedy action from Q-value estimates
        # teammate method: forward(state)  [not forward_state]
        q_values = self.model.forward(state)
        if q_values is None:
            # forward() is still a stub — fall back to random
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(np.array(q_values, dtype=np.float32)))

    # -----------------------------------------------------------------
    # Epsilon decay
    # -----------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """
        Multiplicatively decay epsilon, clamped to EPSILON_MIN.

        Called once at the end of each episode to gradually shift the
        policy from exploration toward exploitation.
        """
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # -----------------------------------------------------------------
    # Batch training
    # -----------------------------------------------------------------
    def _train_on_batch(self, experiences: list) -> float:
        """
        Compute DQN targets and train the network on a mini-batch.

        For each experience (s, a, r, s', done):
            target = r                          if done
            target = r + gamma * max(Q(s'))     otherwise

        Calls model.backward(state, action, reward, next_state, done)
        directly, as that is the interface exposed by the teammate's model.

        Args:
            experiences: List of (state, action, reward, next_state, done).

        Returns:
            Average loss over the batch (0.0 if model returns None stubs).
        """
        total_loss: float = 0.0
        loss_count: int = 0

        for state, action, reward, next_state, done in experiences:
            # Compute the DQN target to guide what backward() should minimise
            if not done:
                q_next = self.model.forward(next_state)
                if q_next is not None:
                    q_next_arr = np.array(q_next, dtype=np.float32)
                    reward = reward + GAMMA * float(np.max(q_next_arr))

            # Train on this transition; teammate signature:
            #   backward(state, action, reward, next_state, done)
            loss_val = self.model.backward(
                state, action, reward, next_state, done
            )

            if loss_val is not None:
                total_loss += float(loss_val)
                loss_count += 1

        return total_loss / loss_count if loss_count > 0 else 0.0

    # -----------------------------------------------------------------
    # Single episode
    # -----------------------------------------------------------------
    def run_episode(self) -> tuple[float, float]:
        """
        Execute a single training episode.

        Steps:
            1. Reset the environment (env.reset()) for a fresh episode start.
            2. Loop until the episode terminates:
                a. Select an action (epsilon-greedy) via model.forward().
                b. Execute the action via env.step(action).
                   Return order: next_state, reward, done
                c. Wrap list states as np.float32 arrays.
                d. Store the transition in replay memory.
                e. If enough experiences exist, sample a mini-batch
                   and train via model.backward() per sample.
                f. Accumulate reward and loss.
            3. Decay epsilon.

        Returns:
            A tuple of (total_episode_reward, average_episode_loss).
        """
        # env.reset() returns list[float] — wrap to np.ndarray
        state: np.ndarray = np.array(self.env.reset(), dtype=np.float32)
        done: bool = False

        episode_reward: float = 0.0
        episode_losses: list[float] = []

        while not done:
            # --- Forward pass & action selection --------------------------
            action: int = self.select_action(state)

            # --- Environment interaction ----------------------------------
            # teammate step() return order: next_state, reward, done
            next_state_raw, reward, done = self.env.step(action)
            next_state: np.ndarray = np.array(next_state_raw, dtype=np.float32)

            # --- Store experience -----------------------------------------
            self.memory.store(state, action, float(reward), next_state, bool(done))

            # --- Train on a mini-batch if sufficient data is available ----
            if len(self.memory) >= BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                loss: float = self._train_on_batch(experiences)
                episode_losses.append(loss)

            # --- Accumulate metrics ---------------------------------------
            episode_reward += float(reward)

            # --- Advance to next state ------------------------------------
            state = next_state

        # Decay exploration rate at the end of the episode
        self.decay_epsilon()

        # Compute average loss (0.0 if no training steps occurred)
        avg_loss: float = (
            float(np.mean(episode_losses)) if episode_losses else 0.0
        )

        return episode_reward, avg_loss

    # -----------------------------------------------------------------
    # Full training run
    # -----------------------------------------------------------------
    def train(self) -> None:
        """
        Run the complete training loop over all episodes, log metrics,
        save model weights, and export history to CSV files.
        """
        logger.info("=" * 60)
        logger.info("Training started")
        logger.info(
            "Episodes: %d | Batch: %d | Memory: %d | Gamma: %.2f",
            EPISODES,
            BATCH_SIZE,
            MEMORY_SIZE,
            GAMMA,
        )
        logger.info(
            "Epsilon: %.2f → %.2f (decay %.4f)",
            EPSILON,
            EPSILON_MIN,
            EPSILON_DECAY,
        )
        logger.info("=" * 60)

        episode: int = 0
        try:
            for episode in range(1, EPISODES + 1):
                episode_reward, avg_loss = self.run_episode()

                # Record metrics
                self.reward_history.append(episode_reward)
                self.loss_history.append(avg_loss)

                # Log progress every episode
                logger.info(
                    "Episode %4d/%d | Reward: %8.2f | Avg Loss: %10.6f | "
                    "Epsilon: %.4f",
                    episode,
                    EPISODES,
                    episode_reward,
                    avg_loss,
                    self.epsilon,
                )

        except KeyboardInterrupt:
            logger.warning(
                "Training interrupted by user at episode %d.", episode
            )
        except Exception as exc:
            logger.error("Training failed with error: %s", exc, exc_info=True)
            raise

        # --- Post-training: save weights and metrics ----------------------
        self._save_model()
        self._save_metrics()

        logger.info("=" * 60)
        logger.info("Training complete.")
        logger.info("=" * 60)

    # -----------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------
    def _save_model(self) -> None:
        """
        Persist the trained network weights to disk.

        Note: QNetwork does not currently expose save_weights().
        This call is guarded and will log a warning if the method
        is absent, rather than crashing the post-training phase.
        """
        try:
            self.model.save_weights()
            logger.info("Model weights saved successfully.")
        except AttributeError:
            logger.warning(
                "model.save_weights() is not implemented in QNetwork. "
                "Weights were NOT saved to disk."
            )
        except Exception as exc:
            logger.error("Failed to save model weights: %s", exc)

    def _save_metrics(self) -> None:
        """Export reward and loss histories to CSV files via pandas."""
        try:
            reward_df = pd.DataFrame(
                {
                    "episode": range(1, len(self.reward_history) + 1),
                    "total_reward": self.reward_history,
                }
            )
            reward_df.to_csv("reward_history.csv", index=False)
            logger.info("Reward history saved to reward_history.csv")

            loss_df = pd.DataFrame(
                {
                    "episode": range(1, len(self.loss_history) + 1),
                    "average_loss": self.loss_history,
                }
            )
            loss_df.to_csv("loss_history.csv", index=False)
            logger.info("Loss history saved to loss_history.csv")

        except Exception as exc:
            logger.error("Failed to save metrics: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Instantiate the Trainer and launch the training loop."""
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
