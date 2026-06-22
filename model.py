import numpy as np

class QNetwork:
    def __init__(self, input_size=4, hidden_size=5, output_size=4, learning_rate=0.01):

        # 1. The Dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # 2. The Weight Matrices (The "Synapses")
        # We use np.random.uniform to create small random numbers between -1 and 1.
        # W1 connects the 4 inputs to the 5 hidden neurons (Shape: 4x5)
        self.W1 = np.random.randn(self.input_size, self.hidden_size)*0.1
        
        # W2 connects the 5 hidden neurons to the 4 output actions (Shape: 5x4)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)*0.1

        # 3. Memory for the Backward Pass
        # We create empty variables here so the network can "remember" its math 
        # during the forward pass, which we will need later to calculate the errors.
        self.state = None
        self.z1 = None
        self.a1 = None
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))
    
    def forward(self, state):
        """
        This is the forward pass which outputs the Q-value prediction for the given state , 
        input will be the 1 D array from the environment and output will be the 1 D array of Q values for each action. 
        """
        # 1. THE SHAPE TRAP: 
        # Forces a 1D array from SurvivalEnv into a 2D array [1, input_size].
        # If train.py sends a batch [32, input_size], it leaves it perfectly alone.
        self.state = np.atleast_2d(state).astype(np.float32)
        
        # 2. HIDDEN LAYER: Linear Transformation
        # Z1 = (State • W1) + b1 
        self.z1 = np.dot(self.state, self.W1) + self.b1
        
        # 3. ACTIVATION: Hyperbolic Tangent (tanh)
        # Adds non-linearity, squishing values between -1 and 1
        self.a1 = np.tanh(self.z1)
        
        # 4. OUTPUT LAYER: Raw Q-Values
        # Z2 = (A1 • W2) + b2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # 5. ALIGNMENT RETURN:
        # If train.py is just asking for a single move (gameplay), flatten it 
        # so np.argmax() works cleanly without shape errors.
        if self.state.shape[0] == 1:
            return self.z2[0]
            
        # If train.py sent a batch of 32 states for learning, return the 2D matrix.
        return self.z2
    
    def backward(self, target_q, current_q):
        """
        Executes Backpropagation using the Chain Rule to calculate gradients,
        clips them for stability, and updates the weights.
        """
        # 1. Calculate the Raw Error
        # Positive error means we guessed too low. Negative means we guessed too high.
        error = target_q - current_q

        error = np.atleast_2d(error)

        # ==========================================
        # 2. OUTPUT LAYER GRADIENTS (W2, b2)
        # ==========================================
        # Since our output layer has no activation function (it is purely linear), 
        # the derivative of Z2 is just the error itself.
        dZ2 = error
        
        # dW2 = Transpose(A1) • dZ2
        dW2 = np.dot(self.a1.T, dZ2)
        # db2 = Sum of dZ2 across the batch
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # ==========================================
        # 3. HIDDEN LAYER GRADIENTS (W1, b1)
        # ==========================================
        # Push the error backward through W2
        dA1 = np.dot(dZ2, self.W2.T)
        
        # Derivative of tanh activation: dZ1 = dA1 * (1 - A1^2)
        dZ1 = dA1 * (1.0 - np.power(self.a1, 2))
        
        # dW1 = Transpose(State) • dZ1
        dW1 = np.dot(self.state.T, dZ1)
        # db1 = Sum of dZ1 across the batch
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # ==========================================
        # 4. GRADIENT CLIPPING (The Safety Net)
        # ==========================================
        # Prevents the "Exploding Gradient" problem inherent in Q-Learning
        dW2 = np.clip(dW2, -1.0, 1.0)
        db2 = np.clip(db2, -1.0, 1.0)
        dW1 = np.clip(dW1, -1.0, 1.0)
        db1 = np.clip(db1, -1.0, 1.0)

        # ==========================================
        # 5. THE OPTIMIZER (Vanilla Gradient Descent)
        # ==========================================
        # We use += because our error formula was (Target - Current)
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2
    
    def save_weights(self, filename="brain_weights.npy"):
        """Saves the matrices to the hard drive so the AI doesn't lose its memory."""
        with open(filename, 'wb') as f:
            np.save(f, self.W1)
            np.save(f, self.b1)
            np.save(f, self.W2)
            np.save(f, self.b2)

    def load_weights(self, filename="brain_weights.npy"):
        """Loads the matrices for test.py to use for the presentation video."""
        with open(filename, 'rb') as f:
            self.W1 = np.load(f)
            self.b1 = np.load(f)
            self.W2 = np.load(f)
            self.b2 = np.load(f)