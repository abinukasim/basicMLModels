import pickle

import numpy as np

# Initialize random parameters for the three-layer neural network
np.random.seed(0)  # For reproducibility

parameters = {
    "theta1": np.random.randn(3, 5) / np.sqrt(3),
    "b1": np.zeros(5),
    "theta2": np.random.randn(5, 5) / np.sqrt(5),
    "b2": np.zeros(5),
    "theta3": np.random.randn(5, 3) / np.sqrt(5),
    "b3": np.zeros(3),
}

# Save parameters to a pickle file
with open("./nn_param_new.pickle", "wb") as file:
    pickle.dump(parameters, file)

print("New network parameters saved to 'nn_param_new.pickle'.")
