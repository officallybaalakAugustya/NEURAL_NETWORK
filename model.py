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
        self.W1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        
        # W2 connects the 5 hidden neurons to the 4 output actions (Shape: 5x4)
        self.W2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))

        # 3. Memory for the Backward Pass
        # We create empty variables here so the network can "remember" its math 
        # during the forward pass, which we will need later to calculate the errors.
        self.state = None
        self.z1 = None
        self.a1 = None
     
    


    
