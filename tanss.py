import collections
import os
import os.path as osp
import tensorflow as tf
from modules import *
from wheel import *
from args import *


class TANSS:
    def __init__(self, n_train):
        self.class_emb = tf.placeholder(tf.float32, [None, args.dim_cls_emb], name="class_emb")
        self.class_emb_all = tf.placeholder(tf.float32, [None, args.dim_cls_emb], name="all_class_emb")
        self.label = tf.placeholder(tf.float32, [None, args.n_class], name="label")
        self.image = tf.placeholder(tf.float32, [None, args.dim_image], name="image")
        self.text = tf.placeholder(tf.float32, [None, args.dim_text], name="text")

        self.training = tf.placeholder("bool", [], name="training")
        self.lr = tf.placeholder(tf.float32, [], name="lr")

        self.emb_pool = tf.placeholder(tf.float32, [None, args.dim_emb], name="emb_pool")
        self.cls_emb_pool = tf.placeholder(tf.float32, [None, args.dim_cls_emb], name="cls_emb_pool")
        self.sim_mat = tf.placeholder(tf.float32, [None, n_train], name="sim_mat")

        self._build_model()
        self._add_loss()
        self._add_optim()

        self.metric_list = ["mAP"]
        self.record = collections.defaultdict(list)

    def _build_model(self):
        print("build model")

        print("- label net")
        print("-- class emb of a batch for 1st & 3rd term, i.e. Eq(2) & Eq(5)")
        self.emb_lb = label_generator(self.class_emb, self.training, args.dim_emb)
        self.reg_cls_emb_lb, self.cls_pred_lb = label_regressor(
            self.emb_lb, self.training, args.dim_cls_emb, args.n_class, args.multi_label)
        print("-- class emb of both S & U for 2nd term, i.e. Eq(4)")
        self.emb_lb_all = label_generator(self.class_emb_all, self.training, args.dim_emb)
        self.reg_cls_emb_lb_all, _ = label_regressor(
            self.emb_lb_all, self.training, args.dim_cls_emb, args.n_class, args.multi_label)

        print("- image net")
        self.emb_im = image_generator(self.image, self.training, args.dim_emb)
        self.reg_cls_emb_im, self.cls_pred_im = image_regressor(
            self.emb_im, self.training, args.dim_cls_emb, args.n_class, args.multi_label)

        print("- text net")
        self.emb_tx = text_generator(self.text, self.training, args.dim_emb)
        self.reg_cls_emb_tx, self.cls_pred_tx = text_regressor(
            self.emb_tx, self.training, args.dim_cls_emb, args.n_class, args.multi_label)

        print("- image discriminator")
        self.isfrom_im = image_discriminator(self.emb_im, self.training)
        self.isfrom_lb_im = image_discriminator(self.emb_lb, self.training)

        print("- text discriminator")
        self.isfrom_tx = text_discriminator(self.emb_tx, self.training)
        self.isfrom_lb_tx = text_discriminator(self.emb_lb, self.training)

    def _add_loss(self, reduce_fn=tf.reduce_mean):
        print("add loss")

        print("- adversarial loss")
        with tf.name_scope("loss_adv"):
            print("-- image v.s. label")
            # loss_adv_im_lb = reduce_fn(tf.math.square(self.isfrom_im - tf.zeros_like(self.isfrom_im)))
            # loss_adv_lb_im = reduce_fn(tf.math.square(self.isfrom_lb_im - tf.ones_like(self.isfrom_lb_im)))
            loss_adv_im_lb = reduce_fn(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.isfrom_im, labels=tf.zeros_like(self.isfrom_im)))
            loss_adv_lb_im = reduce_fn(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.isfrom_lb_im, labels=tf.ones_like(self.isfrom_lb_im)))
            print("-- text v.s. label")
            # loss_adv_tx_lb = reduce_fn(tf.math.square(self.isfrom_tx - tf.zeros_like(self.isfrom_tx)))
            # loss_adv_lb_tx = reduce_fn(tf.math.square(self.isfrom_lb_tx - tf.ones_like(self.isfrom_lb_tx)))
            loss_adv_tx_lb = reduce_fn(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.isfrom_tx, labels=tf.zeros_like(self.isfrom_tx)))
            loss_adv_lb_tx = reduce_fn(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.isfrom_lb_tx, labels=tf.ones_like(self.isfrom_lb_tx)))
            print("-- sum up")
            self.loss_adv = (loss_adv_im_lb + loss_adv_tx_lb + loss_adv_lb_im + loss_adv_lb_tx) / 4.0

        self.summary_adv = tf.summary.merge([
            tf.summary.scalar("loss_adv", self.loss_adv),
            tf.summary.scalar("loss_adv_im_lb", loss_adv_im_lb),
            tf.summary.scalar("loss_adv_tx_lb", loss_adv_tx_lb),
            tf.summary.scalar("loss_adv_lb_im", loss_adv_lb_im),
            tf.summary.scalar("loss_adv_lb_tx", loss_adv_lb_tx),
        ])

        # if args.multi_label:
        #     clf_criterion = tf.nn.sigmoid_cross_entropy_with_logits
        # else:  # single label
        #     clf_criterion = tf.nn.softmax_cross_entropy_with_logits_v2

        print("- label net")
        with tf.name_scope("loss_lb"):
            print("-- pairwise loss")
            # loss_pair_lb = struct_loss(self.emb_lb, self.emb_pool, self.sim_mat, reduce_fn=reduce_fn)
            # loss_pair_lb = struct_loss(self.emb_lb, self.emb_pool, self.sim_mat, reduce_fn=tf.nn.l2_loss)
            loss_pair_lb = struct_loss(self.emb_lb, self.emb_pool, self.sim_mat, reduce_fn=tf.reduce_sum)
            print("-- cycle consistency loss")
            # loss_cyc_lb = reduce_fn(tf.math.square(self.class_emb - self.reg_cls_emb_lb))
            loss_cyc_lb = reduce_fn(tf.math.square(self.class_emb_all - self.reg_cls_emb_lb_all))
            print("-- classification loss")
            loss_clf_lb = reduce_fn(tf.math.square(self.label - self.cls_pred_lb))
            # loss_clf_lb = tf.nn.l2_loss(self.label - self.cls_pred_lb)
            # loss_clf_lb = reduce_fn(clf_criterion(labels=self.label, logits=self.cls_pred_lb))
            print("-- sum up")
            self.loss_lb = args.alpha * loss_pair_lb + args.beta * loss_cyc_lb + \
                args.gamma * loss_clf_lb

        self.summary_lb = tf.summary.merge([
            tf.summary.scalar("loss_lb", self.loss_lb),
            tf.summary.scalar("loss_pair_lb", loss_pair_lb),
            tf.summary.scalar("loss_cyc_lb", loss_cyc_lb),
            tf.summary.scalar("loss_clf_lb", loss_clf_lb),
            # tf.summary.scalar("label_adv_lb_im", loss_adv_lb_im),
            # tf.summary.scalar("label_adv_lb_tx", loss_adv_lb_tx),
        ])

        print("- image net")
        with tf.name_scope("loss_im"):
            print("-- pairwise loss on f")
            # loss_pair_f_im = struct_loss(self.emb_im, self.emb_pool, self.sim_mat, reduce_fn=reduce_fn)
            # loss_pair_f_im = struct_loss(self.emb_im, self.emb_pool, self.sim_mat, reduce_fn=tf.nn.l2_loss)
            loss_pair_f_im = struct_loss(self.emb_im, self.emb_pool, self.sim_mat, reduce_fn=tf.reduce_sum)
            print("-- pairwise loss on regressed e")
            # loss_pair_e_im = struct_loss(self.reg_cls_emb_im, self.cls_emb_pool, self.sim_mat, reduce_fn=reduce_fn)
            # loss_pair_e_im = struct_loss(self.reg_cls_emb_im, self.cls_emb_pool, self.sim_mat, reduce_fn=tf.nn.l2_loss)
            loss_pair_e_im = struct_loss(self.reg_cls_emb_im, self.cls_emb_pool, self.sim_mat, reduce_fn=tf.reduce_sum)
            print("-- classification loss")
            loss_clf_im = reduce_fn(tf.math.square(self.label - self.cls_pred_im))
            # loss_clf_im = tf.nn.l2_loss(self.label - self.cls_pred_im)
            # loss_clf_im = reduce_fn(clf_criterion(labels=self.label, logits=self.cls_pred_im))
            print("-- adv loss")
            loss_adv_im = reduce_fn(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.isfrom_im, labels=tf.ones_like(self.isfrom_im)))
            print("-- sum up")
            self.loss_im = args.alpha * loss_pair_f_im + args.beta * loss_pair_e_im + \
                args.gamma * loss_clf_im + loss_adv_im

        self.summary_im = tf.summary.merge([
            tf.summary.scalar("loss_im", self.loss_im),
            tf.summary.scalar("loss_pair_f_im", loss_pair_f_im),
            tf.summary.scalar("loss_pair_e_im", loss_pair_e_im),
            tf.summary.scalar("loss_clf_im", loss_clf_im),
            tf.summary.scalar("loss_adv_im", loss_adv_im),
        ])

        print("- text net")
        with tf.name_scope("loss_tx"):
            print("-- pairwise loss on f")
            # loss_pair_f_tx = struct_loss(self.emb_tx, self.emb_pool, self.sim_mat, reduce_fn=reduce_fn)
            # loss_pair_f_tx = struct_loss(self.emb_tx, self.emb_pool, self.sim_mat, reduce_fn=tf.nn.l2_loss)
            loss_pair_f_tx = struct_loss(self.emb_tx, self.emb_pool, self.sim_mat, reduce_fn=tf.reduce_sum)
            print("-- pairwise loss on regressed e")
            # loss_pair_e_tx = struct_loss(self.reg_cls_emb_tx, self.cls_emb_pool, self.sim_mat, reduce_fn=reduce_fn)
            # loss_pair_e_tx = struct_loss(self.reg_cls_emb_tx, self.cls_emb_pool, self.sim_mat, reduce_fn=tf.nn.l2_loss)
            loss_pair_e_tx = struct_loss(self.reg_cls_emb_tx, self.cls_emb_pool, self.sim_mat, reduce_fn=tf.reduce_sum)
            print("-- classification loss")
            loss_clf_tx = reduce_fn(tf.math.square(self.label - self.cls_pred_tx))
            # loss_clf_tx = tf.nn.l2_loss(self.label - self.cls_pred_tx)
            # loss_clf_tx = reduce_fn(clf_criterion(labels=self.label, logits=self.cls_pred_tx))
            print("-- adv loss")
            loss_adv_tx = reduce_fn(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.isfrom_tx, labels=tf.ones_like(self.isfrom_tx)))
            print("-- sum up")
            self.loss_tx = args.alpha * loss_pair_f_tx + args.beta * loss_pair_e_tx + \
                args.gamma * loss_clf_tx + loss_adv_tx

        self.summary_tx = tf.summary.merge([
            tf.summary.scalar("loss_tx", self.loss_tx),
            tf.summary.scalar("loss_pair_f_tx", loss_pair_f_tx),
            tf.summary.scalar("loss_pair_e_tx", loss_pair_e_tx),
            tf.summary.scalar("loss_clf_tx", loss_clf_tx),
            tf.summary.scalar("loss_adv_tx", loss_adv_tx),
        ])

    def _add_optim(self):
        print("add optim")

        print("- optimizer")
        optim = tf.train.AdamOptimizer(args.lr)#, beta1=0.5, beta2=0.999)
        optim_adv = tf.train.AdamOptimizer(args.lr)#, beta1=0.5, beta2=0.999)
        # optim_lb = tf.train.AdamOptimizer(self.lr)#, beta1=0.5, beta2=0.999)
        # optim_im = tf.train.AdamOptimizer(self.lr)#, beta1=0.5, beta2=0.999)
        # optim_tx = tf.train.AdamOptimizer(self.lr)#, beta1=0.5, beta2=0.999)
        # optim_adv = tf.train.AdamOptimizer(self.lr)#, beta1=0.5, beta2=0.999)

        print("- var list")
        var_lb, var_im, var_tx, var_adv = [], [], [], []
        for v in tf.trainable_variables():
            if ("LabelGenerator" in v.name) or ("LabelRegressor" in v.name):
                var_lb.append(v)
            elif ("ImageGenerator" in v.name) or ("ImageRegressor" in v.name):
                var_im.append(v)
            elif ("TextGenerator" in v.name) or ("TextRegressor" in v.name):
                var_tx.append(v)
            elif ("ImageDiscriminator" in v.name) or ("TextDiscriminator" in v.name):
                var_adv.append(v)
        print("#var:", len(var_lb), len(var_im), len(var_tx), len(var_adv))

        print("- train op")
        print("-- label net")
        # grad_lb = optim.compute_gradients(self.loss_lb)
        # self.train_lb = optim.apply_gradients(grad_lb)
        self.train_lb = optim.minimize(self.loss_lb, var_list=var_lb)
        _grad_lb = tf.gradients(self.loss_lb, var_lb)
        self.summary_grad_lb = tf.summary.merge([
            tf.summary.histogram("grad/label/{}".format(_g.name), _g) for _g in _grad_lb])

        print("-- image net")
        # grad_im = optim.compute_gradients(self.loss_im)
        # self.train_im = optim.apply_gradients(grad_im)
        self.train_im = optim.minimize(self.loss_im, var_list=var_im)
        _grad_im = tf.gradients(self.loss_im, var_im)
        self.summary_grad_im = tf.summary.merge([
            tf.summary.histogram("grad/label/{}".format(_g.name), _g) for _g in _grad_im])

        print("-- text net")
        # grad_tx = optim.compute_gradients(self.loss_tx)
        # self.train_tx = optim.apply_gradients(grad_tx)
        self.train_tx = optim.minimize(self.loss_tx, var_list=var_tx)
        _grad_tx = tf.gradients(self.loss_tx, var_tx)
        self.summary_grad_tx = tf.summary.merge([
            tf.summary.histogram("grad/label/{}".format(_g.name), _g) for _g in _grad_tx])

        print("-- discriminator")
        # grad_adv = optim.compute_gradients(self.loss_adv)
        # self.train_adv = optim.apply_gradients(grad_adv)
        self.train_adv = optim_adv.minimize(self.loss_adv, var_list=var_adv)
        _grad_adv = tf.gradients(self.loss_adv, var_adv)
        self.summary_grad_adv = tf.summary.merge([
            tf.summary.histogram("grad/label/{}".format(_g.name), _g) for _g in _grad_adv])
