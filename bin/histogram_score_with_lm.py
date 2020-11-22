import kenlm
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import binarize
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

"""
Most of the highest probability images amongst the dataset are "1". This is the same thing described in 
Krusinga, R., Shah, S., Zwicker, M., Goldstein, T., & Jacobs, D. (2019). Understanding the (un) interpretability of 
natural image distributions using generative models. arXiv preprint arXiv:1901.01499.:

"...if we train a GAN on MNIST digits and then test on new MNIST digits, all the most likely digits are simple 1s. If we 
take all the 1s out of the training set, then when we test on the full set of MNIST digits including 1s, the 1s are 
still the most likely, even though the GAN never saw them during training"

The histogram of log probabilities in this paper (figure 4) also look like the histogram of scores plotted here.

"""

if __name__ == '__main__':

    lm = kenlm.Model("../out/kenlm/mnist_binarized_n60.klm")

    mnist = fetch_openml('mnist_784', data_home="../out")

    binarize(mnist.data, copy=False)

    samples_and_scores = []

    for x in tqdm(mnist.data):
        samples_and_scores.append((x, lm.score(" ".join([str(int(i)) for i in x]), bos=False, eos=False)))

    scores = [s[1] for s in samples_and_scores]
    plt.hist(scores, 50, density=True, facecolor='g', alpha=0.75)
    plt.show()

    samples_and_scores = sorted(samples_and_scores, key=lambda v: v[1])

    n_x = 10
    n_y = 10
    top = samples_and_scores[-(n_x*n_y):]
    bottom = samples_and_scores[:(n_x*n_y)]

    f, axarr = plt.subplots(n_x, n_y)
    idx = 0
    for i in range(n_x):
        for j in range(n_y):
            axarr[i, j].imshow(np.array(top[idx][0]).reshape((28, 28)), cmap='gray')
            idx += 1

    plt.show()

    f, axarr = plt.subplots(n_x, n_y)
    idx = 0
    for i in range(n_x):
        for j in range(n_y):
            axarr[i, j].imshow(np.array(bottom[idx][0]).reshape((28, 28)), cmap='gray')
            idx += 1

    plt.show()
