import copy
import math

import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from pydantic import BaseModel
from pydantic_numpy.typing import NpNDArray

BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SET_PERCENT = 0.2
TEST_SET_PERCENT = 0.2
EPOCHS_PATIENCE = 5
LR = 0.001
M_BETA = 0.9
V_BETA = 0.99
LR_DECAY_RATE = 0.001
HIDDEN_LAYERS = 2
HIDDEN_LAYER_NEURONS = 128
OUTPUT_LAYER_NEURONS = 10

class NeuralNetResult(BaseModel):
    weights: list[NpNDArray]
    biases: list[NpNDArray]
    loss_over_epochs: NpNDArray
    accuracy_over_epochs: NpNDArray

def inference(weights: list, biases: list, input: np.ndarray) -> np.ndarray:
    out = input
    layers = len(weights)

    for j in range(layers):
        input = out
        out = input @ weights[j].T + biases[j][np.newaxis, :]

        if j == layers - 1:
            out = softmax(out)
        else:
            out = np.maximum(0, out)

    return out

def neural_net(trainig_data: np.ndarray) -> NeuralNetResult:
    """Returns loss and validation loss over steps. (loss type, step)"""
    N = trainig_data.shape[0]

    validation_set_size = math.floor(N * VALIDATION_SET_PERCENT)
    validation_set = trainig_data[:validation_set_size]
    validation_set_y = validation_set[:, -1].astype(np.int64)

    test_set_size = math.floor(N * TEST_SET_PERCENT)
    test_set = trainig_data[validation_set_size:(validation_set_size + test_set_size)]
    test_set_y = test_set[:, -1].astype(np.int64)

    trainig_data = trainig_data[validation_set_size + test_set_size:]
    training_set_size = N - (validation_set_size + test_set_size)

    hidden_layer_neurons = HIDDEN_LAYER_NEURONS
    hidden_layer_count = HIDDEN_LAYERS

    feature_count = trainig_data.shape[1] - 1 # -1 since the last column is y
    x = trainig_data[:, :feature_count]
    y = trainig_data[:, -1].astype(np.int64)
    rng = default_rng(67)
    weights = []
    biases = []
    m_weights = []
    m_biases = []
    v_weights = []
    v_biases = []

    best_weights = []
    best_biases = []
    loss_values = []
    validation_loss_values = []
    accuracy_values = []
    best_accuracy = 0
    smallest_validation_loss_step = 0
    smallest_validation_loss = np.finfo(np.float64).max

    patience_count = 0

    # Initialization
    for i in range(hidden_layer_count):
        inputs = hidden_layer_neurons if i > 0 else feature_count
        weights.append(rng.standard_normal((hidden_layer_neurons, inputs)) * np.sqrt(2.0 / inputs))
        biases.append(np.zeros(hidden_layer_neurons, dtype=np.float64))

        v_weights.append(np.zeros((hidden_layer_neurons, inputs)))
        v_biases.append(np.zeros(hidden_layer_neurons, dtype=np.float64))

        m_weights.append(np.zeros((hidden_layer_neurons, inputs)))
        m_biases.append(np.zeros(hidden_layer_neurons, dtype=np.float64))

    weights.append(rng.standard_normal((OUTPUT_LAYER_NEURONS, hidden_layer_neurons)) * np.sqrt(2.0 / hidden_layer_neurons))
    biases.append(np.zeros(OUTPUT_LAYER_NEURONS, dtype=np.float64))

    v_weights.append(np.zeros((OUTPUT_LAYER_NEURONS, hidden_layer_neurons)))
    v_biases.append(np.zeros(OUTPUT_LAYER_NEURONS, dtype=np.float64))
    m_weights.append(np.zeros((OUTPUT_LAYER_NEURONS, hidden_layer_neurons)))
    m_biases.append(np.zeros(OUTPUT_LAYER_NEURONS, dtype=np.float64))

    # Training

    steps_per_epoch = math.ceil(training_set_size / BATCH_SIZE)
    steps = EPOCHS * steps_per_epoch

    print(f"Learning Rate: {LR}")
    print(f"Learning Rate Decay: {LR_DECAY_RATE}")
    print("----------")
    print(f"Dataset Size: {training_set_size}")
    print(f"Validationset Size: {validation_set_size}")
    print(f"Testset Size: {test_set_size}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Epochs: {EPOCHS} Patience: {EPOCHS_PATIENCE}")
    print(f"Steps per Epoch: {steps_per_epoch}")
    print(f"Max steps: {steps}")

    epoch_loss = 0
        
    _, out = forward_pass(x, weights, biases)
    l = loss(y, out)
    print(f"Initial loss: {l}")

    for i in range(steps):
        batch_index = i % steps_per_epoch
        bottom = batch_index * BATCH_SIZE
        top = min(bottom + BATCH_SIZE, training_set_size)
        batch_x = x[bottom:top, :]
        batch_y = y[bottom:top]
        one_hot_y = np.eye(OUTPUT_LAYER_NEURONS)[batch_y]

        # Forward Pass
        z_layers, out = forward_pass(batch_x, weights, biases)

        # Backward Pass
        a_in_grad = None
        lr = LR / (1 + LR_DECAY_RATE * i)

        for li in reversed(range(hidden_layer_count + 1)):
            if li == hidden_layer_count:
                z_grad = (softmax(z_layers[li]) - one_hot_y) / batch_x.shape[0]
            else:
                z_grad = relu_gradient(z_layers[li]) * a_in_grad

            # if li == 1:
            #     h = 1e-6
            #     prev = weights[li][1, 2]
            #     weights[li][1, 2] -= h
            #     gc_out1 = inference(weights, biases, batch_x)
            #     weights[li][1, 2] = prev + h
            #     gc_out2 = inference(weights, biases, batch_x)
            #     weights[li][1, 2] = prev
            #
            #     grad_test = (loss(batch_y, gc_out2) - loss(batch_y, gc_out1)) / (2 * h)

            b_grad = z_grad.sum(axis=0)
            m_biases[li] = M_BETA * m_biases[li] + (1 - M_BETA) * b_grad
            v_biases[li] = V_BETA * v_biases[li] + (1 - V_BETA) * np.square(b_grad)

            m_biases_unbiased = m_biases[li] / (1 - np.pow(M_BETA, i + 1))
            v_biases_unbiased = v_biases[li] / (1 - np.pow(V_BETA, i + 1))
            biases[li] -= lr * (m_biases_unbiased / (np.sqrt(v_biases_unbiased) + 1e-8))

            if li > 0:
                a_prev = np.maximum(0, z_layers[li - 1])
                a_in_grad = z_grad @ weights[li]
            else:
                a_prev = batch_x

            w_grad = z_grad.T @ a_prev
            # if li == 1:
            #     passed = w_grad[8, 8] == grad_test
            #     if not passed:
            #         print(f"--- {i} Numeric: {grad_test}  Backprop: {w_grad[8, 8]}")

            m_weights[li] = M_BETA * m_weights[li] + (1 - M_BETA) * w_grad
            v_weights[li] = V_BETA * v_weights[li] + (1 - V_BETA) * np.square(w_grad)

            m_weights_unbiased = m_weights[li] / (1 - np.pow(M_BETA, i + 1))
            v_weights_unbiased = v_weights[li] / (1 - np.pow(V_BETA, i + 1))
            weights[li] -= lr * (m_weights_unbiased / (np.sqrt(v_weights_unbiased) + 1e-8))

        # New Loss
        out = np.squeeze(out)
        l = loss(batch_y, out)
        epoch_loss += l

        if batch_index == steps_per_epoch -1:
            loss_values.append(epoch_loss / steps_per_epoch)
            epoch_loss = 0

            validation_out = inference(weights, biases, validation_set[:, :-1])
            validation_l = loss(validation_set_y, validation_out)
            validation_loss_values.append(validation_l)

            predictions = np.argmax(validation_out, axis=1)
            correct_predictions = np.count_nonzero(predictions == validation_set[:, -1])
            accuracy = correct_predictions / validation_set.shape[0]

            accuracy_values.append(accuracy)

            if validation_l < smallest_validation_loss:
                smallest_validation_loss_step = i + 1
                smallest_validation_loss = validation_l
                best_biases = copy.deepcopy(biases)
                best_weights = copy.deepcopy(weights)
                patience_count = 0
                best_accuracy = accuracy
            else:
                patience_count += 1

            if patience_count >= EPOCHS_PATIENCE:
                print(f"\n\nFINISHED TRAINING - It's step {i}. Using weights from step {smallest_validation_loss_step}\n")
                break

            rng.shuffle(trainig_data)
            y = trainig_data[:, -1].astype(np.int64)

    biases = best_biases
    weights = best_weights
    print(f"Best validation loss at step {smallest_validation_loss_step} and epoch {smallest_validation_loss_step / steps_per_epoch}: {smallest_validation_loss}")
    print(f"Best accuracy: {best_accuracy}")
    print(f"Training Loss: {loss_values[-1]}")

    test_out = inference(weights, biases, test_set[:, :-1])
    test_l = loss(test_set_y, test_out)

    predictions = np.argmax(test_out, axis=1)
    correct_predictions = np.count_nonzero(predictions == test_set[:, -1])
    accuracy = correct_predictions / test_set.shape[0]

    print(f"Test Loss: {test_l} | Test Acc.: {accuracy}")

    loss_over_steps = np.array([loss_values, validation_loss_values])
    accuracy_over_epochs = np.array(accuracy_values)
    return NeuralNetResult(weights=weights, biases=biases, loss_over_epochs=loss_over_steps, accuracy_over_epochs=accuracy_over_epochs)

def forward_pass(x: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    out = x
    layers = len(weights)
    z_layers = []

    for j in range(layers):
        input = out
        out = input @ weights[j].T + biases[j][np.newaxis, :]

        z_layers.append(out.copy())

        if j == layers - 1:
            out = softmax(out)
        else:
            out = np.maximum(0, out)

    return z_layers, out

# Categorial Cross Entropy
def loss(gt: np.ndarray, out: np.ndarray) -> float:
    epsilon = 1e-15
    out = np.clip(out, epsilon, 1 - epsilon)
    y = np.eye(OUTPUT_LAYER_NEURONS)[gt]
    return -1 * (y * np.log(out)).sum(axis=1).mean()

def relu_gradient(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0, 0, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(z):
    max_z = np.max(z, axis=1, keepdims=True)
    shift_z = z - max_z
    exp_z = np.exp(shift_z)
    total = np.sum(exp_z, axis=1, keepdims=True)
    return exp_z / total

def logit(x):
    return np.log(x / (1-x))

def main():
    mnist = fetch_openml('mnist_784')
    df: pd.DataFrame = mnist.frame

    npdata = df.select_dtypes(include=['int64', 'category']).to_numpy(dtype=np.float64)

    x = npdata[:, :-1]
    x /= x.max()
    print(x.max())

    #
    # mean = features.mean(axis=0)
    # std = features.std(axis=0)
    #
    # features -= mean
    # features /= std
    #
    result = neural_net(npdata)
    # biases = result.biases
    # weights = result.weights
    l = result.loss_over_epochs
    acc = result.accuracy_over_epochs

    # biases[0] = biases[0] - (weights[0] @ (mean / std))
    # weights[0] /= std

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
    epochs = np.arange(0, l.shape[1])
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('loss')
    axes[0].plot(epochs, l[0, :], color="blue")
    axes[0].plot(epochs, l[1, :], color="green")

    epochs = np.arange(0, acc.shape[0])
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('accuracy')
    axes[1].plot(epochs, acc, color="green")

    plt.show()


if __name__ == "__main__":
    main()
