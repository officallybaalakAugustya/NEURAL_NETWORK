import random
import logging
from collections import deque

import numpy as np
import pandas as pd

from environment import SurvivalEnv
from model import QNetwork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EPISODES: int = 1000
BATCH_SIZE: int = 32
MEMORY_SIZE: int = 10_000

GAMMA: float = 0.99

EPSILON: float = 1.0
EPSILON_MIN: float = 0.05
EPSILON_DECAY: float = 0.995

NUM_ACTIONS: int = 4


class ReplayMemory:

    def __init__(self, capacity: int = MEMORY_SIZE) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = BATCH_SIZE) -> list:
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} experiences from a buffer "
                f"of size {len(self.buffer)}."
            )
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class Trainer:

    def __init__(self) -> None:
        self.env = SurvivalEnv()
        self.model = QNetwork()
        self.memory = ReplayMemory(capacity=MEMORY_SIZE)
        self.epsilon: float = EPSILON
        self.reward_history: list[float] = []
        self.loss_history: list[float] = []

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        q_values = self.model.forward(state)
        if q_values is None:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(np.array(q_values, dtype=np.float32)))

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def _train_on_batch(self, experiences: list) -> float:
        total_loss: float = 0.0

        for state, action, reward, next_state, done in experiences:
           # model forward pass to get current Q-values
            current_q = self.model.forward(state)
            
            # Create a copy to act as our target
            target_q = np.copy(current_q)
            
            # 2. The Bellman Equation (Calculate the target)
            if done:
                # If the agent died, the reward is final
                target_q[action] = reward
            else:
                # If alive, add the maximum possible future reward
                next_q = self.model.forward(next_state)
                target_q[action] = reward + GAMMA * float(np.max(next_q))

            # 3. math to update the model weights based on the difference between target and current Q-values
            self.model.backward(target_q, current_q)

            # 4. Calculate Mean Squared Error for your loss_history logs
            loss_val = float(np.mean((target_q - current_q) ** 2))
            total_loss += loss_val

        return total_loss / len(experiences)

    def run_episode(self) -> tuple[float, float]:
        state: np.ndarray = np.array(self.env.reset(), dtype=np.float32)
        done: bool = False
        episode_reward: float = 0.0
        episode_losses: list[float] = []

        while not done:
            action: int = self.select_action(state)

            next_state_raw, reward, done = self.env.step(action)
            next_state: np.ndarray = np.array(next_state_raw, dtype=np.float32)

            self.memory.store(state, action, float(reward), next_state, bool(done))

            if len(self.memory) >= BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                loss: float = self._train_on_batch(experiences)
                episode_losses.append(loss)

            episode_reward += float(reward)
            state = next_state

        self.decay_epsilon()

        avg_loss: float = (
            float(np.mean(episode_losses)) if episode_losses else 0.0
        )

        return episode_reward, avg_loss

    def train(self) -> None:
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
            "Epsilon: %.2f -> %.2f (decay %.4f)",
            EPSILON,
            EPSILON_MIN,
            EPSILON_DECAY,
        )
        logger.info("=" * 60)   

        record_score : float = 0.0 

        episode: int = 0
        try:
            for episode in range(1, EPISODES + 1):
                episode_reward, avg_loss = self.run_episode()

                self.reward_history.append(episode_reward)
                self.loss_history.append(avg_loss)
                  
                  # logic to save the model if a new record score is achieved
                if episode_reward > record_score:
                    record_score = episode_reward
                    self._save_model()  # Overwrite the file with the new smartest brain
                    logger.info(f"🏆 New Record Score: {record_score}! Brain saved.")

                logger.info(
                    "Episode %4d/%d | Reward: %8.2f | Avg Loss: %10.6f | Epsilon: %.4f",
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

       
        self._save_metrics()

        logger.info("=" * 60)
        logger.info("Training complete.")
        logger.info("=" * 60)

    def _save_model(self) -> None:
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


def main() -> None:
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
