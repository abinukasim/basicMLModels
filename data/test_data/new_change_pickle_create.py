import pickle

import numpy as np

np.random.seed(0)

changes = {
    "theta1": np.random.randn(3, 5) / np.sqrt(3),
    "b1": np.zeros(5),
    "theta2": np.random.randn(5, 5) / np.sqrt(5),
    "b2": np.zeros(5),
    "theta3": np.random.randn(5, 3) / np.sqrt(5),
    "b3": np.zeros(3),
}

with open("./nn_change_new.pickle", "wb") as file:
    pickle.dump(changes, file)

print("New network parameters saved to 'nn_changes_new.pickle'.")
