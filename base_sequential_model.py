import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf


class BaseSequentialModel:
    """
    An abstract base class for sequential models (for tensorflow)

    This class provides the foundational structure for building and training
    sequential models, including functionalities for saving model weights,
    tracking loss history, and generating training callbacks.

    Attributes:
        vocab_size (int): The size of the vocabulary, representing the total number of unique tokens.
        max_input_len (int): The maximum length of input sequences.
        model (tf.keras.Model): The Keras model instance (to be defined in subclasses).
        model_name (str): The name of the model.
        loss_history (dict): A dictionary to store training loss history.
        hyper_params (dict): A dictionary to store hyperparameters for the model training.
    """

    def __init__(self, vocab_size, max_input_len):
        """
        Initializes the BaseSequentialModel with specified vocabulary size and maximum input length.

        Args:
            vocab_size (int): The size of the vocabulary, representing the total number of unique tokens.
            max_input_len (int): The maximum length of input sequences that the model will process.
        """

        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.model = None
        self.model_name = ""
        self.loss_history = None
        self.hyper_params = {}

    def save_model_path(self):
        """
        Returns the file path for saving the model weights.

        Returns:
            str: The path to save the model weights, formatted as "model_weights/{model_name}_weights.keras".
        """

        return f"model_weights/{self.model_name}_weights.keras"

    def save_losses_path(self):
        """
        Returns the file path for saving the loss history.

        Returns:
            str: The path to save the loss history, formatted as "model_weights/{model_name}_losses.json".
        """
        return f"model_weights/{self.model_name}_losses.json"

    def get_callbacks(self):
        """
        Creates and returns a list of training callbacks.

        This includes callbacks for saving the model weights and reducing the learning rate on plateau.

        Returns:
            list: A list of Keras callbacks for model training.
        """

        os.makedirs("model_weights", exist_ok=True)
        save_model_path = self.save_model_path()
        return [
            tf.keras.callbacks.ModelCheckpoint(
                save_model_path, monitor="loss", save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.2, patience=1, min_lr=0.001
            ),
        ]

    def train(self, x, y, train_from_scratch=False):
        """
        Trains the sequential model on the provided data.

        This method attempts to load existing model weights and loss history if they exist.
        If loading fails or if specified, the model will be trained from scratch.

        Args:
            x (array-like): The input data for training.
            y (array-like): The target data for training.
            train_from_scratch (bool): A flag indicating whether to train the model from scratch.
                                        Defaults to False.
        """

        save_model_path = self.save_model_path()
        save_losses_path = self.save_losses_path()

        if train_from_scratch is False and os.path.exists(save_model_path):
            try:
                self.model.load_weights(save_model_path)
                if os.path.exists(save_losses_path):
                    with open(save_losses_path, "r") as f:
                        self.loss_history = json.load(f)
                print(f"Loaded saved {self.model_name} model and weights.")
            except:
                print(
                    "Could not load pre-trained model possibly due to a mismatch in model architecture. "
                    + "Reverting to training model from scratch."
                )
                self.train(x, y, True)

        else:
            print(f"Training {self.model_name} model from scratch...")
            batch_size = self.hp["batch_size"]
            epochs = self.hp["epochs"]
            callbacks = self.get_callbacks()
            y_onehot = tf.one_hot(y, depth=self.vocab_size)
            y_onehot = tf.squeeze(y_onehot, axis=1)
            full_history = self.model.fit(
                x,
                y_onehot,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )
            print(f"Saved {self.model_name} model weights to {save_model_path}")

            self.loss_history = {"losses": full_history.history["loss"]}
            with open(save_losses_path, "w") as f:
                json.dump(self.loss_history, f)
            print(f"Saved {self.model_name} model loss history to {save_losses_path}")

    def plot_loss(self):
        """
        Plots the training loss history.

        If no training history is available, a message is printed prompting the user to train the model first.
        """

        if self.loss_history is None:
            print("No training history available. Train the model first.")
            return

        losses = self.loss_history["losses"]
        plt.figure(figsize=(8, 5))
        plt.plot(losses, "b-")
        plt.title(f"{self.model_name} Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(len(losses)))
        plt.tight_layout()
        plt.show()
