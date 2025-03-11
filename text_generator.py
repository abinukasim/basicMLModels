import sys

import numpy as np
from rich.console import Console


class TextGenerator:
    """Generates text using trained models."""

    def __init__(self, char_indices, indices_char, max_input_len):
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.max_input_len = max_input_len

    def sample(self, preds, temperature=0.5) -> int:
        """Sample next character index based on predictions."""
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate(self, model, seed_text, length=150, temperature=0.5):
        """Generate text continuation from seed text."""
        seed_text = (
            " " * (self.max_input_len - len(seed_text)) + seed_text
            if len(seed_text) < self.max_input_len
            else seed_text[-self.max_input_len :]
        )

        generated = ""
        print(f"-------------------- {model.model_name} Model --------------------")
        print("Prompt: " + seed_text)
        print("Model: ", end="")

        for _ in range(length):
            x_pred = np.zeros((1, self.max_input_len))
            for t, char in enumerate(seed_text):
                x_pred[0, t] = self.char_indices[char]

            preds = model.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, temperature)
            next_char = self.indices_char[next_index]

            generated += next_char
            seed_text = seed_text[1:] + next_char
            print(next_char, end="")
        print()
