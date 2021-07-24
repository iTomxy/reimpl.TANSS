import numpy as np
import tensorflow as tf


def one_hot(L, n_class):
    """sparse class ID -> one-hot label vector"""
    assert 1 == L.ndim, "not sparse label"
    I = np.eye(n_class)
    return I[L]


def hamming(X, Y=None):
    if Y is None:
        Y = X
    K = tf.cast(tf.shape(X)[1], "float32")
    D = (K - tf.matmul(X, tf.transpose(Y)))
    return tf.clip_by_value(D, 0, K)


def cos(X, Y=None):
    """cosine of every (Xi, Yj) pair
    X, Y: (n, dim)
    """
    X_n = tf.math.l2_normalize(X, axis=1)
    if (Y is None) or (X is Y):
        return tf.matmul(X_n, tf.transpose(X_n))
    Y_n = tf.math.l2_normalize(Y, axis=1)
    _cos = tf.matmul(X_n, tf.transpose(Y_n))
    return tf.clip_by_value(_cos, -1, 1)


def euclidean(A, B=None, sqrt=False):
    if (B is None) or (B is A):
        aTb = tf.matmul(A, tf.transpose(A))
        aTa = bTb = tf.linalg.diag_part(aTb)
    else:
        aTb = tf.matmul(A, tf.transpose(B))
        aTa = tf.linalg.diag_part(tf.matmul(A, tf.transpose(A)))
        bTb = tf.linalg.diag_part(tf.matmul(B, tf.transpose(B)))

    D = aTa[:, None] - 2.0 * aTb + bTb[None, :]
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    D = tf.maximum(D, 0.0)

    if sqrt:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(D, 0.0), "float32")
        D = D + mask * 1e-16
        D = tf.math.sqrt(D)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        D = D * (1.0 - mask)

    return D


def struct_loss(X, Y, S, coef=0.5, reduce_fn=tf.reduce_mean):
    """sum_ij { - [ s_ij * theta_ij - log(1 + exp(theta_ij)) ]}
    = sum_ij { (1 - s_ij) * theta_ij - sigmoid(theta_ij) }
    = sum_ij { log[1 + epx(- |theta_ij|)] + max(0, theta_ij) - s_ij * theta_ij }
    """
    theta = coef * tf.matmul(X, tf.transpose(Y))
    loss = tf.log(1 + tf.exp(- tf.abs(theta))) + tf.maximum(theta, 0) - S * theta
    loss = reduce_fn(loss)
    return loss


def focal_struct_loss(xTy, S, focal_index):
    """struct loss x focal loss
    - S: {0, 1}
    """
    focal_p = tf.math.sigmoid(xTy)
    focal_pos = (1 - focal_p) ** focal_index
    focal_neg = focal_p ** focal_index
    kernel = tf.log(1 + tf.exp(- tf.abs(xTy))) + tf.maximum(xTy, 0) - S * xTy
    loss = tf.where(S > 0, focal_pos, focal_neg) * kernel
    return tf.nn.l2_loss(loss) / 2


def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1., 1.)
