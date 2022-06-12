import matplotlib.pyplot as plt
import numpy as np


def visualize_z_2D(experiment, z, labels, label_list):
    colors = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "brown",
        "fuchsia",
        "grey",
        "olive",
        "lightblue",
    ]
    plt.figure(figsize=(10, 10))
    i = 0
    color_dict = {}
    for p, l in zip(z, labels):
        l = round(l, 2)
        # l = np.argmax(l_arr)
        if l not in color_dict:
            color_dict[l] = i
            i += 1
        plt.scatter(p[0], p[1], marker="${}$".format(label_list[l]), c=colors[color_dict[l]])
    plt.show()
    experiment.log_figure(figure_name="visualize z 2D", figure=plt)


def visualize_z_label(model, dataloader, device):
    colors = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "brown",
        "fuchsia",
        "grey",
        "olive",
        "lightblue",
    ]
    count = 20
    array = []
    while count > 0:
        for x, t in dataloader:
            x = x.to(device)
            y, z = model(x)
            i = 0
            color_dict = {}
            for p, l in zip(z.cpu().detach().numpy(), t.cpu().detach().numpy()):
                l = round(l, 2)
                if l not in color_dict:
                    color_dict[l] = i
                    i += 1
                plt.scatter(p[0], p[1], marker="x", c=colors[color_dict[l]])
            break
        count -= 1
        print(count)
    plt.savefig("result/visualization.png")
    plt.show()