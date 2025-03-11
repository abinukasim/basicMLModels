import pickle

import numpy as np

# Initialize random gradients for the new three-layer neural network
np.random.seed(0)  # For reproducibility

# Create dLoss corresponding to the dimensions of the new network layers
dLoss = {
    "theta1": np.random.randn(3, 5),
    "b1": np.random.randn(5),
    "theta2": np.random.randn(5, 5),
    "b2": np.random.randn(5),
    "theta3": np.random.randn(5, 3),
    "b3": np.random.randn(3),
}

# Save the new dLoss to a pickle file
with open("./dLoss_new.pickle", "wb") as file:
    pickle.dump(dLoss, file)

print("New dLoss saved to 'dLoss_new.pickle'.")
