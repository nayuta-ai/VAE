import matplotlib.pyplot as plt
import numpy as np
import torch


def generate(model, distribution, device, experiment):
    img_size = 28
    num_img = 10
    img_size_space = img_size + 2
    matrix_img = np.zeros((img_size_space * num_img, img_size_space * num_img))

    z_1 = torch.linspace(-3, 3, num_img)
    z_2 = torch.linspace(-3, 3, num_img)

    for i, z1 in enumerate(z_1):
        for j, z2 in enumerate(z_2):
            x = torch.tensor([float(z1), float(z2)], device=device)

            if distribution == "Bern":
                img = model.decoder(x)
            else:
                img = model.reparametrize(model.decoder(x))
            img = img.view(-1, 28, 28)
            img = img.squeeze().detach().cpu().numpy()
            top = i * img_size_space
            left = j * img_size_space
            matrix_img[top: top + img_size, left: left + img_size] = img

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix_img.tolist(), cmap="Greys_r")
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    plt.show()
    experiment.log_figure(figure_name="visualize latent", figure=plt)


def random_generate(model, distribution, latent_dim, device, experiment):
    img_size = 28
    num_img = 10
    img_size_space = img_size + 2
    matrix_img = np.zeros((img_size_space * num_img, img_size_space * num_img))

    for i in range(num_img):
        for j in range(num_img):
            z = torch.randn(latent_dim).to(device)

            if distribution == "Bern":
                img = model.decoder(z)
            else:
                img = model.reparametrize(model.decoder(z))
            img = img.view(-1, 28, 28)
            img = img.squeeze().detach().cpu().numpy()
            top = i * img_size_space
            left = j * img_size_space
            matrix_img[top: top + img_size, left: left + img_size] = img

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix_img.tolist(), cmap="Greys_r")
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    plt.show()
    experiment.log_figure(figure_name="visualize random latent", figure=plt)
