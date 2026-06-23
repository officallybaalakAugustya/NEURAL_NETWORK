import numpy as np
import os

class QNetwork:
    def __init__(self, input_size=8, hidden_size=16, output_size=4, lr=0.001):
        self.lr = lr
        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, hidden_size))

        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b3 = np.zeros((1, output_size))

        # Adam Optimizer Initialization
        self.beta1 = 0.9    
        self.beta2 = 0.999  
        self.epsilon = 1e-8 
        self.t = 0          
        
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)

    # THE FIX: Leaky ReLU prevents Dead Neurons from the -10 penalties
    def leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, z * alpha)

    def leaky_relu_deriv(self, z, alpha=0.01):
        return np.where(z > 0, 1.0, alpha)

    def forward(self, state):
        self.state = np.atleast_2d(state)
        
        self.z1 = np.dot(self.state, self.W1) + self.b1
        self.a1 = self.leaky_relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.leaky_relu(self.z2)

        # Linear Output Layer (Can output negative Q-values safely)
        self.q_values = np.dot(self.a2, self.W3) + self.b3
        return self.q_values

    def _adam_update(self, param, m, v, grad):
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param, m, v

    def backward(self, target_q, current_q):
        self.t += 1 
        
        delta3 = current_q - target_q
        dW3 = np.dot(self.a2.T, delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = np.dot(delta3, self.W3.T) * self.leaky_relu_deriv(self.z2)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = np.dot(delta2, self.W2.T) * self.leaky_relu_deriv(self.z1)
        dW1 = np.dot(self.state.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        clip_value = 1.0
        for grad in [dW1, db1, dW2, db2, dW3, db3]:
            np.clip(grad, -clip_value, clip_value, out=grad)

        self.W3, self.m_W3, self.v_W3 = self._adam_update(self.W3, self.m_W3, self.v_W3, dW3)
        self.b3, self.m_b3, self.v_b3 = self._adam_update(self.b3, self.m_b3, self.v_b3, db3)
        
        self.W2, self.m_W2, self.v_W2 = self._adam_update(self.W2, self.m_W2, self.v_W2, dW2)
        self.b2, self.m_b2, self.v_b2 = self._adam_update(self.b2, self.m_b2, self.v_b2, db2)
        
        self.W1, self.m_W1, self.v_W1 = self._adam_update(self.W1, self.m_W1, self.v_W1, dW1)
        self.b1, self.m_b1, self.v_b1 = self._adam_update(self.b1, self.m_b1, self.v_b1, db1)

    def save_weights(self, filename="brain_weights.npy"):
        try:
            np.save(filename, {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2, 'W3': self.W3, 'b3': self.b3})
        except Exception as e:
            print("Could not save weights:", e)

    # THE FIX: Added the load_weights function for testing tomorrow
    def load_weights(self, filename="brain_weights.npy"):
        if os.path.exists(filename):
            try:
                data = np.load(filename, allow_pickle=True).item()
                self.W1 = data['W1']
                self.b1 = data['b1']
                self.W2 = data['W2']
                self.b2 = data['b2']
                self.W3 = data['W3']
                self.b3 = data['b3']
                print("Brain successfully loaded.")
            except Exception as e:
                print("Error loading brain:", e)
        else:
            print("No saved brain found. Starting with a fresh brain.")