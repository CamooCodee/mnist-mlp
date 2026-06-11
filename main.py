import copy
import math

import sklearn
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from pydantic import BaseModel
from pydantic_numpy.typing import NpNDArray

N = 2500
BATCH_SIZE = 250
EPOCHS = 200
VALIDATION_SET_PERCENT = 0.2
EPOCHS_PATIENCE = 5
LR = 0.05
M_BETA = 0.9
V_BETA = 0.99
LR_DECAY_RATE = 0.005
HIDDEN_LAYERS = 2
HIDDEN_LAYER_NEURONS = 6

class NeuralNetResult(BaseModel):
    weights: list[NpNDArray]
    biases: list[NpNDArray]
    loss_over_epochs: NpNDArray
    accuracy_over_epochs: NpNDArray

def inference(weights: list, biases: list, input: np.ndarray) -> np.ndarray:
    out = input.T
    layers = len(weights)

    for j in range(layers):
        input = out
        out = weights[j] @ input + biases[j][:, np.newaxis]

        if j == layers - 1:
            out = sigmoid(out)
        else:
            out = np.maximum(0, out)

    return np.squeeze(out)

def neural_net(data: np.ndarray) -> NeuralNetResult:
    """Returns loss and validation loss over steps. (loss type, step)"""
    validation_set_size = math.floor(N * VALIDATION_SET_PERCENT)
    validation_set = data[:validation_set_size]
    data = data[validation_set_size:]
    training_set_size = N - validation_set_size

    hidden_layer_neurons = HIDDEN_LAYER_NEURONS
    hidden_layer_count = HIDDEN_LAYERS

    feature_count = data.shape[1] - 1 # -1 since the last column is y
    x = data[:, :feature_count]
    y = data[:, -1]
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
    smallest_validation_loss = 9999

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

    weights.append(rng.standard_normal((1, hidden_layer_neurons)) * np.sqrt(2.0 / hidden_layer_neurons))
    biases.append(np.array([0], dtype=np.float64))

    v_weights.append(np.zeros((1, hidden_layer_neurons)))
    v_biases.append(np.array([0], dtype=np.float64))
    m_weights.append(np.zeros((1, hidden_layer_neurons)))
    m_biases.append(np.array([0], dtype=np.float64))

    # Training

    steps_per_epoch = math.ceil(training_set_size / BATCH_SIZE)
    steps = EPOCHS * steps_per_epoch

    print(f"Learning Rate: {LR}")
    print(f"Learning Rate Decay: {LR_DECAY_RATE}")
    print("----------")
    print(f"Dataset Size: {training_set_size}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Steps per Epoch: {steps_per_epoch}")
    print(f"Max steps: {steps}")

    epoch_loss = 0

    for i in range(steps):
        batch_index = i % steps_per_epoch
        bottom = batch_index * BATCH_SIZE
        top = min(bottom + BATCH_SIZE, training_set_size)
        batch_x = x[bottom:top, :]
        batch_y = y[bottom:top]

        # Forward Pass
        z_layers, out = forward_pass(batch_x, weights, biases)

        # Backward Pass
        a_in_grad = None
        lr = LR / (1 + LR_DECAY_RATE * i)

        for li in reversed(range(hidden_layer_count + 1)):
            if li == hidden_layer_count:
                z_grad = (sigmoid(z_layers[li]) - batch_y) / batch_x.shape[0]
                z_grad = z_grad.T
            else:
                z_grad = relu_gradient(z_layers[li].T) * a_in_grad

            b_grad = z_grad.sum(axis=0)
            m_biases[li] = M_BETA * m_biases[li] + (1 - M_BETA) * b_grad
            v_biases[li] = V_BETA * v_biases[li] + (1 - V_BETA) * np.square(b_grad)

            m_biases_unbiased = m_biases[li] / (1 - np.pow(M_BETA, i + 1))
            v_biases_unbiased = v_biases[li] / (1 - np.pow(V_BETA, i + 1))
            biases[li] -= lr * (m_biases_unbiased / (np.sqrt(v_biases_unbiased) + 1e-8))

            if li > 0:
                a_prev = np.maximum(0, z_layers[li - 1]).T
                a_in_grad = z_grad @ weights[li]
            else:
                a_prev = batch_x

            w_grad = z_grad.T @ a_prev
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
            validation_l = loss(validation_set[:, -1], validation_out)
            validation_loss_values.append(validation_l)

            correct_predictions = np.count_nonzero((validation_out > 0.5) == validation_set[:, -1])
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
                print(f"EXIT AT: {i} with weights from {smallest_validation_loss_step}")
                break

            rng.shuffle(data)

    biases = best_biases
    weights = best_weights
    print(f"Best validation loss at step {smallest_validation_loss_step} and epoch {smallest_validation_loss_step / steps_per_epoch}: {smallest_validation_loss}")
    print(f"Best accuracy: {best_accuracy}")

    loss_over_steps = np.array([loss_values, validation_loss_values])
    accuracy_over_epochs = np.array(accuracy_values)
    return NeuralNetResult(weights=weights, biases=biases, loss_over_epochs=loss_over_steps, accuracy_over_epochs=accuracy_over_epochs)

def forward_pass(x: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    out = x.T
    layers = len(weights)
    z_layers = []

    for j in range(layers):
        input = out
        out = weights[j] @ input + biases[j][:, np.newaxis]

        z_layers.append(out.copy())

        if j == layers - 1:
            out = sigmoid(out)
        else:
            out = np.maximum(0, out)

    return z_layers, out

# Binary Cross Entropy
def loss(gt: np.ndarray, out: np.ndarray) -> float:
    epsilon = 1e-15
    out = np.clip(out, epsilon, 1 - epsilon)
    return -(gt * np.log(out) + (1 - gt) * np.log(1 - out)).mean()

def relu_gradient(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0, 0, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(x):
    return np.log(x / (1-x))

def main():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))
    data = sklearn.datasets.make_moons(n_samples=N, noise=0.225, random_state=68)
    (x, y) = data
    npdata = np.column_stack(data)

    features = npdata[:, :2]

    mean = features.mean(axis=0)
    std = features.std(axis=0)

    features -= mean
    features /= std

    c: list[str] = []

    for i in range(0, N):
        c.append("red" if y[i] == 0 else "green")

    print(x.shape)


    print("-- Data")
    print(npdata.shape)
    print("--\n\n")

    result = neural_net(npdata)
    biases = result.biases
    weights = result.weights
    l = result.loss_over_epochs
    acc = result.accuracy_over_epochs

    biases[0] = biases[0] - (weights[0] @ (mean / std))
    weights[0] /= std

    epochs = np.arange(0, l.shape[1])
    axes[0, 1].set_xlabel('epochs')
    axes[0, 1].set_ylabel('loss')
    axes[0, 1].plot(epochs, l[0, :], color="blue")
    axes[0, 1].plot(epochs, l[1, :], color="green")

    epochs = np.arange(0, acc.shape[0])
    axes[1, 0].set_xlabel('epochs')
    axes[1, 0].set_ylabel('accuracy')
    axes[1, 0].plot(epochs, acc, color="green")

    x1_min = x[:, 0].min()
    x1_max = x[:, 0].max()
    x2_min = x[:, 1].min()
    x2_max = x[:, 1].max()

    x1 = np.arange(x1_min, x1_max + 0.1, 0.1)
    x2 = np.arange(x2_min, x2_max + 0.1, 0.1)
    x1_contour, x2_contour = np.meshgrid(x1, x2)

    x1_joined = x1_contour.flatten()
    x2_joined = x2_contour.flatten()
    x_grid = np.array([x1_joined, x2_joined]).T
    contour_results = inference(weights, biases, x_grid)

    y_contour = contour_results.reshape(-1, x1_contour.shape[1])

    axes[0, 0].contourf(x1_contour, x2_contour, y_contour, alpha=0.5, cmap="RdYlGn")
    axes[0, 0].scatter(x=x[:, 0], y=x[:, 1], c=c, s=6, alpha=0.5)

    plt.show()


if __name__ == "__main__":
    main()
