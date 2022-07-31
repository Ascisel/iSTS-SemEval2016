import matplotlib.pyplot as plt


def plot_model_loss(losses, filename):
    x_axis = [x + 1 for x in range(len(losses))]
    plt.plot(x_axis, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(filename)
