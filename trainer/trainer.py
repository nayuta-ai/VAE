import numpy as np
import torch
import matplotlib.pyplot as plt


def train(model, dataloader, optimizer, device, iteration, experiment):
    with experiment.train():
        model.train()
        for i in range(iteration):
            losses = []
            for x, t in dataloader:
                x = x.to(device)
                model.zero_grad()
                y = model(x)
                loss = model.loss(x)
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy())
            print("EPOCH: {} loss: {}".format(i, np.average(losses)))
            experiment.log_metric("accuracy", np.average(losses), step=iteration)


def val(model, dataloader, device, experiment):
    fig = plt.figure(figsize=(10, 3))
    with experiment.test():
        model.eval()
        zs = []
        for x, t in dataloader:
            # original
            for i, im in enumerate(x.view(-1, 28, 28).detach().numpy()[:10]):
                ax = fig.add_subplot(3, 10, i+1, xticks=[], yticks=[])
                ax.imshow(im, 'gray')
                experiment.log_image(image_data=im, name="original", step=i)
            x = x.to(device)
            # generate from x
            y, z = model(x)
            zs.append(z)
            y = y.view(-1, 28, 28)
            for i, im in enumerate(y.cpu().detach().numpy()[:10]):
                ax = fig.add_subplot(3, 10, i+11, xticks=[], yticks=[])
                ax.imshow(im, 'gray')
                experiment.log_image(image_data=im, name="generate", step=i)
            # generate from z
            z1to0 = torch.cat([z[1] * (i * 0.1) + z[0] * ((9 - i) * 0.1) for i in range(10)])
            z1to0 = torch.reshape(z1to0, (10, 10))
            y2 = model._decoder(z1to0).view(-1, 28, 28)
            for i, im in enumerate(y2.cpu().detach().numpy()):
                ax = fig.add_subplot(3, 10, i+21, xticks=[], yticks=[])
                ax.imshow(im, 'gray')
                experiment.log_image(image_data=im, name="generated_by_z", step=i)
            break