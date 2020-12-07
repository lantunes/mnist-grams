from sklearn.datasets import fetch_openml
from sklearn.preprocessing import binarize
from tqdm import tqdm
import numpy as np


def reduce(image):
    image = np.array(image, dtype=np.int).reshape((28, 28))

    reduced = []

    for i in list(range(28))[::2]:
        for j in list(range(28))[::2]:
            ixgrid = np.ix_([i, i + 1], [j, j + 1])
            stride = image[ixgrid]
            binary = stride.flatten()
            decimal = int("".join([str(x) for x in binary]), 2)
            reduced.append(decimal)

    reduced_image = np.array(reduced).reshape((14, 14))

    return reduced_image.flatten()


if __name__ == '__main__':

    mnist = fetch_openml('mnist_784', data_home="../out")

    print(mnist.data.shape)

    binarize(mnist.data, copy=False)

    with open("../out/mnist_3_reduced_corpus.txt", "wt") as f:
        for i, x in tqdm(enumerate(mnist.data)):
            if mnist.target[i] is not '3':
                continue
            reduced_x = reduce(x)
            f.write(" ".join([str(int(i)) for i in reduced_x]) + "\n")