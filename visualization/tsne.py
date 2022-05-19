from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize_z(experiment, z, labels):
    colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]
    plt.figure(figsize=(10,10))
    points = TSNE(n_components=2, random_state=0).fit_transform(z)
    for p, l in zip(points, labels):
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l])
    plt.show()
    experiment.log_figure(figure_name="visualize z", figure=plt)