import sklearn
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

def main():
    N = 200
    data = sklearn.datasets.make_moons(n_samples=N, noise=0.2, random_state=67)
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

    plt.scatter(x=npdata[:, 0], y=npdata[:, 1], c=c, s=5)

    print("-- Data")
    print(npdata.shape)
    print("--\n\n")

    neural_net(npdata)

    # plt.show()

def neural_net(data: np.ndarray):
    hidden_layer_neurons = 6
    hidden_layer_count = 2

    feature_count = data.shape[1] - 1 # -1 since the last column is y
    y = data[:, -1]
    rng = default_rng(67)
    weight_layers = []
    bias_layers = []

    for i in range(hidden_layer_count):
        inputs = hidden_layer_neurons if i > 0 else feature_count
        weight_layers.append(rng.uniform(low=-1, high=1, size=(hidden_layer_neurons, inputs)))
        bias_layers.append(rng.uniform(low=-1, high=1, size=hidden_layer_neurons))

    weight_layers.append(rng.uniform(low=-1, high=1, size=(1, hidden_layer_neurons)))
    bias_layers.append(np.array([0.67]))

    for i in range(2000):
        # Forward Pass

        x = data[:, :2]
        out = x.T
        layers = len(weight_layers)
        z_layers = []

        for j in range(layers):
            input = out
            weights = weight_layers[j]
            biases = bias_layers[j]
            out = weights @ input + biases[:, np.newaxis]

            z_layers.append(out.copy())

            if j == layers - 1:
                out = sigmoid(out)
            else:
                out = np.tanh(out)


        # Backward Pass
        
        a_in_grad = np.empty
        lr = 3

        #
        for li in reversed(range(hidden_layer_count + 1)):
            if li == hidden_layer_count:
                z_grad = (sigmoid(z_layers[li]) - y) / data.shape[0]
                z_grad = z_grad.T
            else:
                z_grad = tanh_gradient(z_layers[li].T) * a_in_grad

            b_grad = z_grad.sum(axis=0)
            bias_layers[li] -= b_grad * lr

            if li > 0:
                a_prev = np.tanh(z_layers[li - 1]).T
                a_in_grad = z_grad @ weight_layers[li]
            else:
                a_prev = x

            w_grad = z_grad.T @ a_prev
            weight_layers[li] -= w_grad * lr

        if i % 50 == 0:
            out = np.squeeze(out)
            L = loss(data[:, -1], out)
            print(f"{i}. Loss: {L}")

# Binary Cross Entropy
def loss(gt: np.ndarray, out: np.ndarray) -> float:
    return -(gt * np.log(out) + (1 - gt) * np.log(1 - out)).mean()

def tanh_gradient(v: np.ndarray) -> np.ndarray:
    return (1 - np.square(np.tanh(v)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    main()
