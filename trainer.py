import matplotlib.pyplot as plt
import numpy as np
import torch

from visualization.tsne import visualize_z


def train(
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
            for x, t in dataloader_train:
                x = x.to(device)
                model.zero_grad()
                loss = model.loss(x)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.cpu().detach().numpy())
            print("EPOCH: {} train loss: {}".format(i, np.average(train_losses)))
            torch.save(model.state_dict(), "result/model.pth")
            experiment.log_metric("train_loss", np.average(train_losses), step=i)
            val_losses = []
            model.eval()
            for x, t in dataloader_val:
                x = x.to(device)
                loss = model.loss(x)
                val_losses.append(loss.cpu().detach().numpy())
            print("EPOCH: {} val_loss: {}".format(i, np.average(val_losses)))
            experiment.log_metric("val_loss", np.average(val_losses), step=i)


def test(model, dataloader, device, experiment):
    fig = plt.figure(figsize=(10, 3))
    with experiment.test():
        # zs = []
        for x, t in dataloader:
            # original
            for i, im in enumerate(x.view(-1, 28, 28).detach().numpy()[:10]):
                ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(im, "gray")
                experiment.log_image(image_data=im, name="original", step=i)
            x = x.to(device)
            # generate from x
            y, z = model(x)
            # zs.append(z)
            y = y.view(-1, 28, 28)
            for i, im in enumerate(y.cpu().detach().numpy()[:10]):
                ax = fig.add_subplot(3, 10, i + 11, xticks=[], yticks=[])
                ax.imshow(im, "gray")
                experiment.log_image(image_data=im, name="generate", step=i)
            experiment.log_figure(figure_name="visualization", figure=fig)
            visualize_z(experiment, z.cpu().detach().numpy(), t.cpu().detach().numpy())
            break