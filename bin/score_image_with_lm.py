import kenlm
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import binarize
from random import choice

if __name__ == '__main__':

    lm = kenlm.Model("../out/kenlm/mnist_binarized_n60.klm")

    mnist = fetch_openml('mnist_784', data_home="../out")

    binarize(mnist.data, copy=False)

    x = mnist.data[23]
    score = lm.score(" ".join([str(int(i)) for i in x]), bos=False, eos=False)
    print(score)  # log10 probability of image

    random_x = " ".join(choice('01') for _ in range(784))
    score = lm.score(random_x, bos=False, eos=False)
    print(score)  # log10 probability of image
