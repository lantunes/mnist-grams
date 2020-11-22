from sklearn.datasets import fetch_openml
from sklearn.preprocessing import binarize
from tqdm import tqdm

if __name__ == '__main__':

    mnist = fetch_openml('mnist_784', data_home="../out")

    print(mnist.data.shape)

    binarize(mnist.data, copy=False)

    with open("../out/mnist_corpus.txt", "wt") as f:
        for x in tqdm(mnist.data):
            f.write(" ".join([str(int(i)) for i in x]) + "\n")