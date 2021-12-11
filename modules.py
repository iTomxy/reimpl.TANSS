import tensorflow as tf
from args import *


def label_generator(x, is_train, dim_emb, scope_name="LabelGenerator"):
    """label -> 4096 -> 1024 -> embedding (K)"""
    print("--- label generator ---")
    print("label:", x.shape.as_list())
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc1")
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc1",
            kernel_initializer=tf.keras.initializers.he_normal())
        print("fc1:", x.shape.as_list())
        # x = tf.layers.dropout(x, 0.1, training=is_train)
        x = tf.layers.dense(x, 1024, tf.math.tanh, name="fc2",
            kernel_initializer=tf.keras.initializers.he_normal())
        print("fc2:", x.shape.as_list())
        # x = tf.layers.dropout(x, 0.1, training=is_train)
        emb = tf.layers.dense(x, dim_emb, None, name="emb")
        print("embedding:", emb.shape.as_list())

    return emb


def label_regressor(x, is_train, dim_cls_emb, n_class, multi_label=False, scope_name="LabelRegressor"):
    """embedding (K) -> 1024 -> 4096 -> (class_emb, label_pred)"""
    print("--- label regressor ---")
    print("embedding:", x.shape.as_list())
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # x = tf.layers.dense(x, 1024, tf.math.tanh, name="fc1")
        x = tf.layers.dense(x, 1024, tf.nn.leaky_relu, name="fc1",
            kernel_initializer=tf.initializers.he_normal())
        print("fc1:", x.shape.as_list())
        # x = tf.layers.dropout(x, 0.1, training=is_train)
        # x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc2")
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc2",
            kernel_initializer=tf.initializers.he_normal())
        print("fc2:", x.shape.as_list())
        # x = tf.layers.dropout(x, 0.1, training=is_train)
        # x -> (cls_emb, cls_pred)
        cls_emb = tf.layers.dense(x, dim_cls_emb, None, name="class_emb")
        print("class embedding:", cls_emb.shape.as_list())
        # _act_fn = None
        _act_fn = tf.math.sigmoid if multi_label else tf.nn.softmax
        label_pred = tf.layers.dense(x, n_class, _act_fn, name="label_pred")
        print("label prediction:", label_pred.shape.as_list())

    return cls_emb, label_pred


def image_generator(x, is_train, dim_emb, scope_name="ImageGenerator"):
    """image -> 4096 -> 4096 -> embedding (K)"""
    print("--- image generator ---")
    print("label:", x.shape.as_list())
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # if args.tanh_G:
        #     x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc1",
        #         kernel_initializer=tf.initializers.glorot_normal())
        # else:
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc1",
            kernel_initializer=tf.initializers.he_normal())
        print("fc1:", x.shape.as_list())
        x = tf.layers.dropout(x, 0.1, training=is_train)
        # if args.tanh_G:
        #     x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc2",
        #         kernel_initializer=tf.initializers.glorot_normal())
        # else:
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc2",
            kernel_initializer=tf.initializers.he_normal())
        print("fc2:", x.shape.as_list())
        x = tf.layers.dropout(x, 0.3, training=is_train)
        emb = tf.layers.dense(x, dim_emb, None, name="emb")
        print("embedding:", emb.shape.as_list())

    return emb


def image_regressor(x, is_train, dim_cls_emb, n_class, multi_label=False, scope_name="ImageRegressor"):
    """embedding (K) -> 4096 -> 4096 -> (class_emb, label_pred)"""
    print("--- image regressor ---")
    print("embedding:", x.shape.as_list())
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc1",
        #     kernel_initializer=tf.initializers.glorot_normal())
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc1",
            kernel_initializer=tf.initializers.he_normal())
        print("fc1:", x.shape.as_list())
        # x = tf.layers.dropout(x, 0.1, training=is_train)
        # x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc2",
        #     kernel_initializer=tf.initializers.glorot_normal())
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc2",
            kernel_initializer=tf.initializers.he_normal())
        print("fc2:", x.shape.as_list())
        # x = tf.layers.dropout(x, 0.1, training=is_train)
        # x -> (cls_emb, cls_pred)
        cls_emb = tf.layers.dense(x, dim_cls_emb, None, name="class_emb")
        print("class embedding:", cls_emb.shape.as_list())
        # _act_fn = None
        _act_fn = tf.math.sigmoid if multi_label else tf.nn.softmax
        label_pred = tf.layers.dense(x, n_class, _act_fn, name="label_pred")
        print("label prediction:", label_pred.shape.as_list())

    return cls_emb, label_pred


def text_generator(x, is_train, dim_emb, scope_name="TextGenerator"):
    """text -> 4096 -> 4096 -> embedding (K)"""
    print("--- text generator ---")
    print("text:", x.shape.as_list())
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # if args.tanh_G:
        #     x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc1",
        #         kernel_initializer=tf.initializers.glorot_normal())
        # else:
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc1",
            kernel_initializer=tf.initializers.he_normal())
        print("fc1:", x.shape.as_list())
        x = tf.layers.dropout(x, 0.3, training=is_train)
        # if args.tanh_G:
        #     x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc2",
        #         kernel_initializer=tf.initializers.glorot_normal())
        # else:
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc2",
            kernel_initializer=tf.initializers.he_normal())
        print("fc2:", x.shape.as_list())
        x = tf.layers.dropout(x, 0.3, training=is_train)
        emb = tf.layers.dense(x, dim_emb, None, name="emb")
        print("embedding:", emb.shape.as_list())

    return emb


def text_regressor(x, is_train, dim_cls_emb, n_class, multi_label=False, scope_name="TextRegressor"):
    """embedding (K) -> 4096 -> 4096 -> (class_emb, label_pred)"""
    print("--- text regressor ---")
    print("embedding:", x.shape.as_list())
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc1",
        #     kernel_initializer=tf.initializers.glorot_normal())
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc1",
            kernel_initializer=tf.initializers.he_normal())
        print("fc1:", x.shape.as_list())
        # x = tf.layers.dropout(x, 0.1, training=is_train)
        # x = tf.layers.dense(x, 4096, tf.math.tanh, name="fc2",
        #     kernel_initializer=tf.initializers.glorot_normal())
        x = tf.layers.dense(x, 4096, tf.nn.leaky_relu, name="fc2",
            kernel_initializer=tf.initializers.he_normal())
        print("fc2:", x.shape.as_list())
        # x = tf.layers.dropout(x, 0.1, training=is_train)
        # x -> (cls_emb, cls_pred)
        cls_emb = tf.layers.dense(x, dim_cls_emb, None, name="class_emb")
        print("class embedding:", cls_emb.shape.as_list())
        # _act_fn = None
        _act_fn = tf.math.sigmoid if multi_label else tf.nn.softmax
        label_pred = tf.layers.dense(x, n_class, _act_fn, name="label_pred")
        print("label prediction:", label_pred.shape.as_list())

    return cls_emb, label_pred


def image_discriminator(x, is_train, scope_name="ImageDiscriminator"):
    """embedding (K) -> 4096 -> 2048 -> 1"""
    print("--- image discriminator ---")
    print("embedding:", x.shape.as_list())
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x, 4096, tf.nn.relu, name="fc1",
            kernel_initializer=tf.initializers.he_normal())
        print("fc1:", x.shape.as_list())
        x = tf.layers.dropout(x, 0.1, training=is_train)
        x = tf.layers.dense(x, 2048, tf.nn.relu, name="fc2",
            kernel_initializer=tf.initializers.he_normal())
        print("fc2:", x.shape.as_list())
        x = tf.layers.dropout(x, 0.1, training=is_train)
        y = tf.layers.dense(x, 1, None, name="dis_pred")

    return y


def text_discriminator(x, is_train, scope_name="TextDiscriminator"):
    """embedding (K) -> 4096 -> 2048 -> 1"""
    print("--- text discriminator ---")
    print("embedding:", x.shape.as_list())
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x, 4096, tf.nn.relu, name="fc1",
            kernel_initializer=tf.initializers.he_normal())
        print("fc1:", x.shape.as_list())
        x = tf.layers.dropout(x, 0.1, training=is_train)
        x = tf.layers.dense(x, 2048, tf.nn.relu, name="fc2",
            kernel_initializer=tf.initializers.he_normal())
        print("fc2:", x.shape.as_list())
        x = tf.layers.dropout(x, 0.1, training=is_train)
        y = tf.layers.dense(x, 1, None, name="dis_pred")

    return y
