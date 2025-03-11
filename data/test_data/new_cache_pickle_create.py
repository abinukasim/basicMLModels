import pickle

import numpy as np

# Create dummy cache data for a neural network with new dimensions
np.random.seed(0)

cache_data = {
    "mask": np.random.rand(10, 5),
    "X": np.random.rand(10, 3),
    "o1": np.random.rand(10, 5),  # Example shape (10 samples, 5 units in layer 1)
    "u1": np.random.rand(10, 5),  # Shape should match o1
    "o2": np.random.rand(10, 5),  # Example shape for second layer
    "u2": np.random.rand(10, 5),  # Shape should match o2
    "u3": np.random.rand(
        10, 3
    ),  # Final output layer shape (10 samples, 3 output classes)
}

# Save the cache to a pickle file
with open("./test_compute_gradients_cache_new.pickle", "wb") as file:
    pickle.dump(cache_data, file)

print("New cache data saved to 'test_compute_gradients_cache_new.pickle'.")
