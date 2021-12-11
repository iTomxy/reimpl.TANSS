# reimpl.TANSS

Re-implementation of TANSS [1] based on SSAH [3,4], with TensorFlow 1.12.

Notes that it can NOT reproduce the results on the original paper currently. The best results I can get now is listed in [Current Results](#current-results).

# Data

- wikipedia: [5]
- pascal sentences: [6]

# Current Results

| | S i2t | S t2i | U i2t | U t2i |
| :--: | :--: | :--: | :--: | :--: |
| wikipedia | 0.6154 $\pm$ 0.0596 | 0.9102 $\pm$ 0.0205 | 0.2628 $\pm$ 0.0159 | 0.2581 $\pm$ 0.0136 |
| pascal sentences | 0.7827 $\pm$ 0.0429 | 0.8163 $\pm$ 0.0386 | 0.3461 $\pm$ 0.0202 | 0.3320 $\pm$ 0.0303 |

# Tuning Schemes

There're two reasonable tuning schemes:

1. `zero-shot` mode: modeling zero-shot scenario by dividing the classes in training set into S & U.
2. `LRY` mode: following 刘若愚's idea, simply select a validation set from the training set WITHOUT further dividing the training class set.

Accordingly, the command-line argument `tune` is switched into an `int` variable, with:

- `0` denoting NON-tuning status, *i.e.* normal training & testing
- `1` denoting the zero-shot tuning mode
- `2` denoting the LRY's tuning mode

# References

1. [TCYB 2019 | Ternary Adversarial Networks With Self-Supervision for Zero-Shot Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/8771379)

2. [徐行 Xing Xu](https://interxuxing.github.io/)

3. [CVPR 2018 | Self-Supervised Adversarial Hashing Networks for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/8578544)

4. [lelan-li/SSAH](https://github.com/lelan-li/SSAH)

5. [wikipedia数据集预处理](https://blog.csdn.net/HackerTom/article/details/104491152)

6. [Pascal Sentences数据集预处理](https://blog.csdn.net/HackerTom/article/details/121525787)

7. [刘若愚 Ruoyu Liu](https://liuruoyu.github.io/)
