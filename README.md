# Perceptron Binary Classifier

A simple implementation of a binary classifier using a Perceptron algorithm. This project demonstrates fundamental machine learning concepts, including model training, inference, and evaluation on synthetic and real datasets.

## Project Overview

This classifier is built from scratch using Python and Numpy, without relying on advanced machine learning libraries. It aims to showcase the working principles of a Perceptron, one of the foundational algorithms in machine learning, particularly in binary classification tasks.

The model is tested on:
- **Synthetic data**: Randomly generated features and labels.
- **Iris dataset**: Binary classification version, using only the first two classes.

## Features
- Custom Perceptron training loop
- Evaluation on synthetic and Iris datasets
- Basic performance metric: error rate

## Dependencies
- `numpy`
- `scikit-learn` (for `train_test_split` and data loading)

Install the dependencies with:
```bash
pip install numpy scikit-learn
