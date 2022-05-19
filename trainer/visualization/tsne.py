from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize_z(experiment, z, labels):
    colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]
    plt.figure(figsize=(10,5))
    points = TSNE(n_components=2, random_state=0).fit_transform(z)
    i = 0
    color_dict = {}
    for p, l in zip(points, labels):
        l = round(l, 2)
        if l not in color_dict:
            color_dict[l] = i
            i += 1
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[color_dict[l]])
    plt.show()
    experiment.log_figure(figure_name="visualize z", figure=plt)