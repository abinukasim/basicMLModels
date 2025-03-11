import tensorflow as tf
from base_sequential_model import BaseSequentialModel


class LSTM(BaseSequentialModel):
    def __init__(self, vocab_size, max_input_len):
        """
        Initializes the LSTM model with the specified vocabulary size and maximum input length.

        Args:
            vocab_size (int): The size of the vocabulary, representing the total number of unique characters or tokens.
            max_input_len (int): The maximum length of input sequences that the model will process.

        This class inherits from the abstract class BaseSequentialModel, which provides methods for
        training and evaluation. Additionally, the model name is set to "LSTM".
        """
        super().__init__(vocab_size, max_input_len)
        self.model_name = "LSTM"

    def set_hyperparameters(self):
        """
        Sets the hyperparameters for the LSTM model.

        This method initializes the hyperparameters used for training the LSTM, including:
        - embedding_dim: The dimensionality of the embedding layer.
        - lstm_units: The number of units in the LSTM layer.
        - learning_rate: The learning rate for the optimizer.
        - batch_size: The number of samples per gradient update.
        - epochs: The number of epochs to train the model.

        These hyperparameters are stored in the `hp` attribute for easy access during model training.
        """

        self.hp = {
            "embedding_dim": 256,
            "lstm_units": 128,
            "learning_rate": 0.01,
            "batch_size": 128,
            "epochs": 10,
        }

    def define_model(self):
        """
        Defines a sequential LSTM model for character-level text generation.

        The model consists of:
        1. An Embedding layer that converts input characters to dense vectors of size embedding_dim
            - Each scalar (character) is transformed into a vector of size embedding_dim
            - Input shape: (batch_size, sequence_length), Output shape: (batch_size, sequence_length, embedding_dim)

        2. An LSTM layer for sequence processing
            - Input shape: (batch_size, sequence_length, embedding_dim), Output shape: (batch_size, rnn_units)

        3. A Dense layer that maps to vocabulary size
            - Input shape: (batch_size, rnn_units), Output shape: (batch_size, vocab_size)

        4. A Softmax activation layer to output probability distribution over vocabulary
            - Input shape: (batch_size, vocab_size), Output shape: (batch_size, vocab_size)

        Hint:
            Use self.hp for getting embedding dimension & LSTM units
            Use self.vocab_size for vocabulary size

        Returns:
            None. Sets self.model with the defined tf.keras.Sequential model

        # TODO: Implement this function
        """
        raise NotImplementedError()

    def build_model(self):
        """
        Compiles and builds model.
        """
        learning_rate = self.hp["learning_rate"]

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy")
        self.model.build((None, self.vocab_size))
