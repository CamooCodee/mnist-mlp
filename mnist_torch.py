import math
from sklearn.datasets import fetch_openml
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

LR_DECAY = 0.0001

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
        

def main():
    mnist = fetch_openml('mnist_784')
    x_raw = mnist.data.values
    y_raw = mnist.target.to_numpy().astype(np.int64)

    x = torch.tensor(x_raw, dtype=torch.float32) / 255
    y = torch.tensor(y_raw)

    validation_set_size = math.ceil(x.shape[0] * 0.2)
    test_set_size = math.ceil(x.shape[0] * 0.2)
    # train_set_size = x.shape[0] - validation_set_size - test_set_size

    validation_x = x[:validation_set_size]
    validation_y = y[:validation_set_size]
    test_x = x[:test_set_size + validation_set_size]
    test_y = y[:test_set_size + validation_set_size]
    train_x = x[test_set_size + validation_set_size:]
    train_y = y[test_set_size + validation_set_size:]

    print(train_x.shape)
    print(train_y.shape)

    model = Model()
    loss = nn.CrossEntropyLoss()
    adam = Adam(model.parameters(), lr=0.005)
    scheduler = LambdaLR(adam, lr_lambda=lambda step: 1 / (1 + step * LR_DECAY))

    for _ in range(150):
        logits = model(train_x)
        output = loss(logits, train_y)

        output.backward()
        adam.step()
        scheduler.step()
        adam.zero_grad()

    logits = model(validation_x)
    output = loss(logits, validation_y)

    print(f"Validation loss: {output}")


if __name__ == "__main__":
    main()
