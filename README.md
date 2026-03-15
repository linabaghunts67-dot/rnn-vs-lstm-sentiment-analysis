# RNN vs LSTM for Sentiment Analysis

This project investigates two important concepts in deep learning for sequence models:

1. The **vanishing gradient problem** in vanilla RNNs
2. The impact of **stop-word removal** on deep learning models

The experiments compare the performance of a standard RNN and an LSTM on sentiment classification.

---

## Project Goals

The goal is to experimentally validate two theoretical claims:

1. **Standard RNNs suffer from vanishing gradients**, which makes them forget early words in long sequences.
2. **Removing stop words can harm deep learning models**, because words like "not" carry important semantic meaning.

---

## Models Implemented

### LSTM Model

Uses PyTorch's `nn.LSTM`.

Architecture:

Embedding → LSTM → Linear → Sigmoid

---

### Vanilla RNN Model

Uses PyTorch's `nn.RNN`.

Architecture:

Embedding → RNN → Linear → Sigmoid

---

## Experiment 1: RNN vs LSTM

The models were trained on movie review sentiment classification.

Expected result:

| Model | Accuracy |
|------|------|
| RNN | ~60% |
| LSTM | ~85% |

The RNN struggles because gradients shrink during backpropagation through time.

---

## Experiment 2: Stop Word Removal

Original review:
