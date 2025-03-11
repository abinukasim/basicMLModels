import numpy as np
import requests
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_housing_dataset():
    dataset = fetch_california_housing()  # load the dataset
    x, y = dataset.data, dataset.target
    y = y.reshape(-1, 1)
    perm = np.random.RandomState(seed=3).permutation(x.shape[0])[:500]
    x = x[perm]
    y = y[perm]

    index_array = np.argsort(y.flatten())
    x, y = x[index_array], y[index_array]

    values_per_list = len(y) // 3
    list1 = y[:values_per_list]
    list2 = y[values_per_list : 2 * values_per_list]
    list3 = y[2 * values_per_list :]
    label_mapping = {
        tuple(value): label
        for label, value_list in enumerate([list1, list2, list3])
        for value in value_list
    }
    updated_values = [label_mapping[tuple(value)] for value in y]
    num_classes = len(set(updated_values))
    one_hot_encoded = np.eye(num_classes)[updated_values]
    y = np.array(one_hot_encoded)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=1
    )  # split data

    x_scale = MinMaxScaler()
    x_train = x_scale.fit_transform(x_train)  # normalize data
    x_test = x_scale.transform(x_test)

    return x_train, y_train, x_test, y_test


def clean_text(text):
    text = text.lower()
    text = " ".join(text.split())
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz0123456789.,!?'\"\n ;:-")
    text = "".join(c for c in text if c in allowed_chars)
    for punct in ".,!?":
        text = text.replace(punct + punct, punct)
    for punct in ".,!?;:":
        text = text.replace(punct + " ", punct)
        text = text.replace(punct, punct + " ")
    for punct in ".,!?;:":
        text = text.replace(" " + punct, punct)
    text = text.replace('" ', '"').replace(' "', '"')
    text = text.replace("' ", "'").replace(" '", "'")
    text = " ".join(text.split())

    return text


def preprocess_text_data(text):
    """Process text data and return relevant components."""
    sequence_len = 30
    sliding_window_step = 1

    text = clean_text(text)

    vocab = sorted(set(text))
    vocab_size = len(vocab)

    # Create character mappings
    char_indices = {c: i for i, c in enumerate(vocab)}
    indices_char = {i: c for i, c in enumerate(vocab)}

    # Segment text into sequences
    sentences = []
    next_chars = []
    for i in range(0, len(text) - sequence_len, sliding_window_step):
        sentences.append(text[i : i + sequence_len])
        next_chars.append(text[i + sequence_len])

    # Convert sequences to numerical vectors
    x = np.zeros((len(sentences), sequence_len), dtype=np.float32)
    y = np.zeros((len(sentences), 1), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t] = char_indices[char]
        y[i] = char_indices[next_chars[i]]
    # Return all necessary componentsx
    return {
        "x": x,
        "y": y,
        "text": text,
        "char_indices": char_indices,
        "indices_char": indices_char,
        "vocab": vocab,
        "vocab_size": vocab_size,
        "sequence_len": sequence_len,
    }
