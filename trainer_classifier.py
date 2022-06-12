import numpy as np
import torch
import torch.nn as nn


def train(
    generater,
    model,
    dataloader_train,
    dataloader_val,
    optimizer,
    device,
    iteration,
    experiment,
):
    with experiment.train():
        for i in range(iteration):
            train_losses = []
            model.train()
            acc = []
            for x, t in dataloader_train:
                x = x.to(device)
                t = t.to(device)
                _, z = generater(x)
                model.zero_grad()
                loss = model.loss(z, t)
                loss.backward()
                optimizer.step()
                acc.append(model.accuracy(z, t))
                train_losses.append(loss.cpu().detach().numpy())
            print("EPOCH: {} train loss: {} train acc: {}".format(
                i, np.average(train_losses), np.average(acc)))
            torch.save(model.state_dict(), "result/model.pth")
            experiment.log_metric("train_loss", np.average(train_losses), step=i)
            experiment.log_metric("train_acc", np.average(acc), step=i)
            val_losses = []
            model.eval()
            acc = []
            for x, t in dataloader_val:
                x = x.to(device)
                t = t.to(device)
                _, z = generater(x)
                loss = model.loss(z, t)
                acc.append(model.accuracy(z, t))
                val_losses.append(loss.cpu().detach().numpy())
            print("EPOCH: {} val_loss: {} val_acc: {}".format(
                i, np.average(val_losses), np.average(acc)))
            experiment.log_metric("val_loss", np.average(val_losses), step=i)
            experiment.log_metric("val_acc", np.average(acc), step=i)

