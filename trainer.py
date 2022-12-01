import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import *


def train(device, train_loader, valid_loader, optimizer, model, epochs, filepath, modelname):

    def train_one_epoch():
        train_running_loss = 0.
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            x1, x2, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(x1, x2).squeeze()
            # Compute the loss and its gradients
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            # Gather data and report
            train_running_loss += loss.item()
        return train_running_loss / (i + 1)  # loss per batch

    for epoch in range(epochs):
        print(f'EPOCH {epoch}:')
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_avg_loss = train_one_epoch()
        # Note the training loss of every epoch
        # We don't need gradients on to do reporting
        model.train(False)

        valid_running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                x1, x2, vlabels = data[0].to(device), data[1].to(device), data[2].to(device)
                voutputs = model(x1, x2).squeeze()
                loss = F.mse_loss(voutputs, vlabels)
                valid_running_loss += loss.item()
            valid_avg_loss = valid_running_loss / (i + 1)
            print(f"LOSS[{modelname}  {filepath}] train: {train_avg_loss} valid: {valid_avg_loss}")