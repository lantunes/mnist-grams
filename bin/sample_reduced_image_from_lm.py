import kenlm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def reconstruct(sample):
    reconstructed = np.zeros((28, 28), dtype=np.int)

    for r in range(14):
        for c in range(14):
            decimal = sample[r][c]
            stride = np.array([int(x) for x in bin(decimal)[2:].zfill(4)]).reshape((2, 2))
            recon_row = 2 * r
            recon_col = 2 * c
            reconstructed[recon_row][recon_col] = stride[0][0]
            reconstructed[recon_row][recon_col + 1] = stride[0][1]
            reconstructed[recon_row + 1][recon_col] = stride[1][0]
            reconstructed[recon_row + 1][recon_col + 1] = stride[1][1]

    return reconstructed


def normalize(probs):
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]


def get_score(lm, pixel, context):
    return lm.score(context + " " + pixel, bos=False, eos=False)


def sample(lm, n):
    ctx = [0] * 10

    keep_last = n
    vocab = [str(i) for i in range(16)]

    while len(ctx) < 196:
        scores = [get_score(lm, i, " ".join([str(x) for x in ctx[-keep_last:]])) for i in vocab]
        weights = [(10 ** i) for i in scores]
        c = np.random.choice(vocab, p=normalize(weights))
        ctx.append(c)

    return " ".join([str(i) for i in ctx])


if __name__ == '__main__':
    n = 50
    lm = kenlm.Model("../out/kenlm/mnist_3_reduced_binarized_n%s.klm" % n)

    samples = []
    for i in tqdm(range(5000)):
        s = sample(lm, n)
        score = lm.score(s, bos=False, eos=False)
        samples.append((s, score))

    samples = sorted(samples, key=lambda v: v[1])

    n_x = 8
    n_y = 8
    best_samples = samples[-(n_x*n_y):]
    # worst_samples = samples[:(n_x*n_y)]
    f, axarr = plt.subplots(n_x, n_y)
    idx = 0
    for i in range(n_x):
        for j in range(n_y):
            image = np.array([int(i) for i in best_samples[idx][0].split(" ")]).reshape((14, 14))
            axarr[i, j].imshow(reconstruct(image), cmap='gray')
            idx += 1

    plt.show()

