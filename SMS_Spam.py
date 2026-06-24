import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ======================================
# LOAD DATASET
# ======================================

df = pd.read_csv(
    r"C:\Sathwik_Projects\ML_Projects\SMS_Spam_Detection\spam.csv",
    encoding="latin-1"
)

# Remove unnecessary columns
df = df.drop(
    ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],
    axis=1
)

# Rename columns
df = df.rename(
    columns={
        "v1": "label",
        "v2": "Text"
    }
)

# Convert labels to numbers
df["label_enc"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print(df.head())

# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    df["Text"],
    df["label_enc"],
    test_size=0.2,
    random_state=42
)

# Convert to NumPy arrays
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# ======================================
# DATASET INFO
# ======================================

avg_words_len = round(
    sum(
        [len(i.split()) for i in df["Text"]]
    ) / len(df["Text"])
)

total_words_length = len(
    set(
        " ".join(df["Text"]).split()
    )
)

print(f"Training Samples : {len(X_train_np)}")
print(f"Average Words    : {avg_words_len}")
print(f"Vocabulary Size  : {total_words_length}")

# ======================================
# TEXT VECTORIZATION
# ======================================

text_vec = TextVectorization(
    max_tokens=total_words_length,
    standardize="lower_and_strip_punctuation",
    output_mode="int",
    output_sequence_length=avg_words_len
)

text_vec.adapt(X_train_np)

# ======================================
# HELPER FUNCTIONS
# ======================================

def compile_and_fit(model, epochs=5):

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train_np,
        y_train_np,
        epochs=epochs,
        validation_data=(X_test_np, y_test_np)
    )

    return history


def get_metrics(model, X, y):

    predictions = np.round(model.predict(X))

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1_score": f1_score(y, predictions)
    }

# ======================================
# MODEL 1 : DENSE + EMBEDDING
# ======================================

input_layer = layers.Input(
    shape=(1,),
    dtype=tf.string
)

x = text_vec(input_layer)

x = layers.Embedding(
    input_dim=total_words_length,
    output_dim=128
)(x)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(
    32,
    activation="relu"
)(x)

output_layer = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model_1 = keras.Model(
    input_layer,
    output_layer,
    name="Dense_Model"
)

print("\nTraining Dense Model...\n")

history_1 = compile_and_fit(model_1)

# ======================================
# MODEL 2 : Bi-LSTM
# ======================================

input_layer = layers.Input(
    shape=(1,),
    dtype=tf.string
)

x = text_vec(input_layer)

x = layers.Embedding(
    input_dim=total_words_length,
    output_dim=128
)(x)

x = layers.Bidirectional(
    layers.LSTM(
        64,
        return_sequences=True
    )
)(x)

x = layers.Bidirectional(
    layers.LSTM(64)
)(x)

x = layers.Flatten()(x)

x = layers.Dropout(0.1)(x)

x = layers.Dense(
    32,
    activation="relu"
)(x)

output_layer = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model_2 = keras.Model(
    input_layer,
    output_layer,
    name="BiLSTM_Model"
)

print("\nTraining Bi-LSTM Model...\n")

history_2 = compile_and_fit(model_2)

# ======================================
# MODEL 3 : UNIVERSAL SENTENCE ENCODER
# ======================================

use_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    trainable=False,
    input_shape=[],
    dtype=tf.string
)

input_layer = layers.Input(
    shape=[],
    dtype=tf.string
)

embedding = layers.Lambda(
    lambda x: use_layer(x),
    output_shape=(512,)
)(input_layer)

x = layers.Dense(
    64,
    activation="relu"
)(embedding)

x = layers.Dropout(0.2)(x)

output_layer = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model_3 = keras.Model(
    input_layer,
    output_layer,
    name="USE_Model"
)

print("\nTraining USE Model...\n")

history_3 = compile_and_fit(model_3)

# ======================================
# EVALUATION
# ======================================

results = {
    "Dense Embedding": get_metrics(
        model_1,
        X_test_np,
        y_test_np
    ),
    "BiLSTM": get_metrics(
        model_2,
        X_test_np,
        y_test_np
    ),
    "USE": get_metrics(
        model_3,
        X_test_np,
        y_test_np
    )
}

print("\nRESULTS\n")

for model_name, scores in results.items():

    print(f"\n{model_name}")

    for metric, value in scores.items():
        print(
            f"{metric}: {value:.4f}"
        )