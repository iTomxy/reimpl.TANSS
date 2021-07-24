import numpy as np
from sklearn.preprocessing import normalize


def sim_mat(label, label_2=None, sparse=False):
    if label_2 is None:
        label_2 = label
    if sparse:
        S = (label[:, np.newaxis] == label_2[np.newaxis, :])
    else:
        S = np.dot(label, label_2.T) > 0
    return S.astype(label.dtype)


def calc_dist_in_batch(dist_fn, A, B=None, threshold=5000):
    if B is None:
        B = A
    if (A.shape[0] < threshold) and (B.shape[0] < threshold):
        return dist_fn(A, B)
    to_transpose = False
    if A.shape[0] < B.shape[0]:
        A, B = B, A
        to_transpose = True
    to_seq = (B.shape[0] > threshold)
    res = np.zeros([A.shape[0], B.shape[0]]).astype(A.dtype)
    for i in range(0, A.shape[0], threshold):
        a = A[i: i + threshold]
        if to_seq:
            for k in range(0, B.shape[0], threshold):
                b = B[k: k + threshold]
                res[i:i+threshold, k:k+threshold] = dist_fn(a, b)
        else:
            res[i:i+threshold] = dist_fn(a, B)
    if to_transpose:
        res = res.T
    return res


def cos(A, B=None):
    """cosine"""
    An = normalize(A, norm='l2', axis=1)
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    Bn = normalize(B, norm='l2', axis=1)
    return np.dot(An, Bn.T)


def hamming(A, B=None):
    """A, B: [None, bit]
    elements in {-1, 1}
    """
    if B is None: B = A
    bit = A.shape[1]
    return (bit - A.dot(B.T)) * 0.5


def euclidean(A, B=None, sqrt=False):
    if (B is None) or (B is A):
        aTb = A.dot(A.T)
        aTa = bTb = np.diag(aTb)
    else:
        aTb = A.dot(B.T)
        aTa = (A * A).sum(1)
        bTb = (B * B).sum(1)
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D


def mAP(Dist, S, k=-1):
    """mean Average Precision
    - Dist: distance matrix
    - S: similarity matrix
    - k: mAP@k, default `-1` means mAP@ALL
    ref:
    - https://blog.csdn.net/HackerTom/article/details/89309665
    """
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    Gnd = S.astype(np.int32)
    gnd_rs = np.sum(Gnd, axis=1)
    Rank = np.argsort(Dist)

    AP = 0.0
    for it in range(n):
        gnd = Gnd[it]
        if 0 == gnd_rs[it]:
            continue
        rank = Rank[it][:k]
        gnd = gnd[rank]
        if (k > 0) and (np.sum(gnd) == 0):
            continue
        pos = np.asarray(np.where(gnd == 1.)) + 1.0
        rel_cnt = np.arange(pos.shape[-1]) + 1.0
        AP += np.mean(rel_cnt / pos)

    return AP / n


def nDCG(Dist, Rel, k=-1):
    """Normalized Discounted Cumulative Gain
    ref: https://github.com/kunhe/TALR/blob/master/%2Beval/NDCG.m
    """
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    G = 2 ** Rel - 1
    D = np.log2(2 + np.arange(k))
    Rank = np.argsort(Dist)

    _NDCG = 0
    for g, rnk in zip(G, Rank):
        dcg_best = (np.sort(g)[::-1][:k] / D).sum()
        if dcg_best > 0:
            dcg = (g[rnk[:k]] / D).sum()
            _NDCG += dcg / dcg_best
    return _NDCG / n
