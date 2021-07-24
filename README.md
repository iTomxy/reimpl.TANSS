Re-implementation of TANSS [1] based on SSAH [3,4].

Notes that this implementation can **NOT** fully reproduce the results of the paper now, with several percents lower.

# Environment

- [Tensorflow 1.12.0](https://hub.docker.com/layers/tensorflow/tensorflow/1.12.0-gpu-py3/images/sha256-413b9533f92a400117a23891d050ab829e277a6ff9f66c9c62a755b7547dbb1e?context=explore)


# Data

I provide the wikipedia data in `data/wikipedia`, along with the splitting indices in `data/wikipedia/disjoint`.

The data have been validated with DADN [7], with which the results of the paper [6] can be reproduced successfully.

See [5] for the details of the wikipedia dataset processing.

# Usage

## tuning

````shell
bash scripts/tune-wiki-disjoint.sh
````

## running

```shell
bash scripts/run-disjoint.sh
```

# Tuning Scheme

There're `2` reasonable tuning schemes:

1. `zero-shot` mode: modeling zero-shot scenario by dividing the classes in training set into S & U.
2. `LRY` mode: following 刘若愚's idea, simply select a validation set from the training set **WITHOUT** further dividing the training class set.

Accordingly, the command-line argument `tune` is switched into an `int` variable, with:

- `0` denoting **NON**-tuning status, *i.e.* normal training & testing
- `1` denoting the zero-shot tuning mode
- `2` denoting the LRY's tuning mode

# References

1. [TCYB 2019 | Ternary Adversarial Networks With Self-Supervision for Zero-Shot Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/8771379)
2. [徐行 Xing Xu](https://interxuxing.github.io/)
3. [CVPR 2018 | Self-Supervised Adversarial Hashing Networks for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/8578544)
4. [lelan-li/SSAH](https://github.com/lelan-li/SSAH)
5. [wikipedia数据集预处理](https://blog.csdn.net/HackerTom/article/details/104491152)
6. [TCSVT 2019 | Zero-Shot Cross-Media Embedding Learning With Dual Adversarial Distribution Network](https://ieeexplore.ieee.org/document/8643797)
7. [PKU-ICST-MIPL/DADN_TCSVT2019](https://github.com/PKU-ICST-MIPL/DADN_TCSVT2019)
8. [刘若愚 Ruoyu Liu](https://liuruoyu.github.io/)
