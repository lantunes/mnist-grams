import kenlm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def normalize(probs):
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]


def get_score(lm, pixel, context):
    return lm.score(context + " " + pixel, bos=False, eos=False)


def sample(lm, n):
    ctx = ["0"] * 10
    ctx = " ".join(ctx)

    keep_last = 2*n - 1

    while len(ctx.replace(" ", "")) < 784:
        score_0 = get_score(lm, "0", ctx[-keep_last:])
        score_1 = get_score(lm, "1", ctx[-keep_last:])

        c = np.random.choice(["0", "1"], p=normalize([10 ** score_0, 10 ** score_1]))
        ctx = ctx + " " + c

    return ctx


if __name__ == '__main__':
    lm = kenlm.Model("../out/kenlm/mnist_binarized_n60.klm")
    n = 60

    samples = []
    for i in tqdm(range(500)):
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
            axarr[i, j].imshow(np.array([int(i) for i in best_samples[idx][0].replace(" ", "")]).reshape((28, 28)), cmap='gray')
            idx += 1

    plt.show()

