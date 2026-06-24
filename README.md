# SMS Spam Detection Using Deep Learning

## Overview

Unwanted spam messages are a common problem in digital communication, often leading to security risks and poor user experience. This project aims to automatically classify SMS messages as either spam or legitimate (ham) using Deep Learning and Natural Language Processing (NLP) techniques.

Multiple neural network architectures were developed and compared, including Dense Embeddings, Bidirectional LSTM (Bi-LSTM), and Universal Sentence Encoder (USE), to identify the most effective approach for SMS spam detection.

---

## Dataset

The dataset contains 5,572 labeled SMS messages categorized as:

- Ham (Legitimate Message)
- Spam

The messages were pre-labeled and used for supervised learning.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- TensorFlow Hub
- Pandas
- NumPy
- Scikit-Learn

---

## Methodology

### Data Preparation

- Removed unnecessary columns from the dataset
- Renamed features for better readability
- Converted text labels into numerical values
- Split the dataset into training and testing sets

### Text Processing

- Applied text vectorization
- Converted SMS messages into numerical representations
- Generated embeddings suitable for deep learning models

### Models Implemented

#### Dense Embedding Model
- Embedding Layer
- Global Average Pooling
- Dense Neural Network

#### Bidirectional LSTM Model
- Embedding Layer
- Two Bidirectional LSTM Layers
- Dropout Regularization
- Dense Output Layer

#### Universal Sentence Encoder (USE)
- Pre-trained Universal Sentence Encoder embeddings
- Dense Neural Network
- Dropout Regularization

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|---------|----------|
| Dense Embedding | 97.85% | 94.33% | 89.26% | 91.72% |
| Bi-LSTM | 98.03% | 95.04% | 89.93% | 92.41% |
| Universal Sentence Encoder | 98.39% | 95.17% | 92.62% | 93.88% |

The Universal Sentence Encoder achieved the best overall performance, providing the highest accuracy and F1-score among all tested models.

---

## Project Structure

```text
SMS-Spam-Detection/
│
├── spam.csv
├── SMS_Spam.py
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/SMS-Spam-Detection.git

cd SMS-Spam-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the project using:

```bash
python SMS_Spam.py
```

The script will:

1. Load and preprocess the dataset.
2. Train all three deep learning models.
3. Evaluate their performance.
4. Display accuracy, precision, recall, and F1-score for comparison.

---

## Key Learnings

- Understanding text vectorization and embeddings for NLP tasks.
- Building and training deep learning models for text classification.
- Comparing traditional embedding approaches with pre-trained language representations.
- Evaluating classification models using multiple performance metrics.

---

## Future Improvements

- Deploy the model as a web application using Streamlit or Flask.
- Experiment with Transformer-based architectures such as BERT.
- Perform hyperparameter tuning for improved performance.
- Extend the system to support multilingual spam detection.

---

## Author

Sathwik Gorrela

B.Tech Mechanical Engineering  
Indian Institute of Technology Madras
