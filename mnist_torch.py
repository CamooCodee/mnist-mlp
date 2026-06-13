import copy
import math
from typing import Mapping
from sklearn.datasets import fetch_openml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

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

def init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        

def main():
    mnist = fetch_openml('mnist_784')
    x_raw = mnist.data.values
    y_raw = mnist.target.to_numpy().astype(np.int64)

    x = torch.tensor(x_raw, dtype=torch.float32) / 255
    y = torch.tensor(y_raw)

    validation_set_size = math.floor(x.shape[0] * 0.2)
    test_set_size = math.floor(x.shape[0] * 0.2)
    # train_set_size = x.shape[0] - validation_set_size - test_set_size

    validation_x = x[:validation_set_size].to("mps")
    validation_y = y[:validation_set_size].to("mps")
    test_x = x[validation_set_size:(validation_set_size + test_set_size)].to("mps")
    test_y = y[validation_set_size:(validation_set_size + test_set_size)].to("mps")
    train_x = x[test_set_size + validation_set_size:]
    train_y = y[test_set_size + validation_set_size:]

    train_set = TensorDataset(train_x, train_y)
    loader = DataLoader(train_set, shuffle=True, batch_size=64)

    LR_DECAY = 0.0001
    NUM_EPOCHS = 100
    PATIENCE_EPOCHS = 6

    best_val_loss = torch.finfo(torch.float32).max
    best_val_loss_epoch = 0
    best_state: Mapping = {}
    patience_counter = 0

    tracked_train_loss = []
    tracked_val_loss = []
    tracked_accuracy = []

    model = Model().to("mps")
    model.apply(init_he)
    loss = nn.CrossEntropyLoss()
    adam = Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.99))
    scheduler = LambdaLR(adam, lr_lambda=lambda step: 1 / (1 + step * LR_DECAY))

    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        training_loss = 0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to("mps")
            batch_y = batch_y.to("mps")

            logits = model(batch_x)
            output = loss(logits, batch_y)

            output.backward()
            adam.step()
            scheduler.step()
            adam.zero_grad()

            training_loss += output.item()

        logits = model(validation_x)
        output = loss(logits, validation_y)

        val_loss = output.item()
        train_loss = training_loss / len(loader)
        tracked_train_loss.append(train_loss)
        tracked_val_loss.append(val_loss)

        correct = (torch.argmax(logits, dim=1) == validation_y).sum().item()
        accuracy = correct / validation_y.shape[0]
        tracked_accuracy.append(accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > PATIENCE_EPOCHS:
            break

    print("\n-- Finished Training --\n")
    print(f"Using state from epoch {best_val_loss_epoch} with a validation loss of {best_val_loss}")

    model.load_state_dict(best_state)

    logits = model(test_x)
    output = loss(logits, test_y)

    print(f"Test loss: {output.item()}")

    correct = (torch.argmax(logits, dim=1) == test_y).sum().item()
    accuracy = correct / test_y.shape[0]

    print(f"Test accuracy: {accuracy}")

    plt.figure(figsize=(18, 10))
    axes = plt.subplot(1, 2, 1)

    axes.set_xlabel("epochs")
    axes.set_ylabel("loss")
    epochs = np.arange(1, len(tracked_train_loss) + 1)
    axes.plot(epochs, np.array(tracked_train_loss), color="palevioletred")
    axes.plot(epochs, np.array(tracked_val_loss), color="dodgerblue")

    axes = plt.subplot(1, 2, 2)

    axes.set_xlabel("epochs")
    axes.set_ylabel("accuracy")
    epochs = np.arange(1, len(tracked_accuracy) + 1)
    axes.plot(epochs, np.array(tracked_accuracy), color="olivedrab")

    plt.show()


if __name__ == "__main__":
    main()
