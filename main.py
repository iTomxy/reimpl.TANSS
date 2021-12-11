import collections
import math
import os
import os.path as osp
import shutil
import random
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import tanss
import wikipedia
import pascal_sentences
# import flickr25k
# import nuswide
# import coco
import evaluate
from wheel import *
from tool import *
from args import *


def init_emb_pool(sess, model, dataset):
    emb_pool, cls_emb_pool = [], []
    loader = dataset.iter_set(dataset.idx_train_s,
        index=False, label=False, image=False, text=False)
    for cls_emb_batch in loader:
        # _emb, _cls_emb = sess.run([model.emb_lb, model.reg_cls_emb_lb],
        #     feed_dict={model.class_emb: cls_emb_batch, model.training: False})
        _emb, _cls_emb = sess.run([model.emb_lb, model.reg_cls_emb_lb],
            feed_dict={model.class_emb: cls_emb_batch, model.training: False})
        emb_pool.append(_emb)
        # cls_emb_pool.append(_cls_emb)
        cls_emb_pool.append(cls_emb_batch)

    emb_pool = np.vstack(emb_pool)
    cls_emb_pool = np.vstack(cls_emb_pool)
    return emb_pool, cls_emb_pool


def gen_feature(sess, model, dataset, indices):
    """label, image fea, text fea"""
    L, Fi, Ft = [], [], []
    loader = dataset.iter_set(indices, index=False, class_emb=False)
    for label, image, text in loader:
        L.append(label)
        emb_im, emb_tx = sess.run([model.emb_im, model.emb_tx],
            feed_dict={model.image: image, model.text: text, model.training: False})
        Fi.append(emb_im)
        Ft.append(emb_tx)

    if args.sparse_label:
        L = np.concatenate(L)
    else:
        L = np.vstack(L)
    Fi = np.vstack(Fi)
    Ft = np.vstack(Ft)
    return L, Fi, Ft


def test(sess, model, dataset):
    res = {}
    qls, qis, qts = gen_feature(sess, model, dataset, dataset.idx_test_s)
    qlu, qiu, qtu = gen_feature(sess, model, dataset, dataset.idx_test_u)
    rls, ris, rts = gen_feature(sess, model, dataset, dataset.idx_ret_s)
    rlu, riu, rtu = gen_feature(sess, model, dataset, dataset.idx_ret_u)

    D_i2t_s = 1 - evaluate.cos(qis, rts)
    D_t2i_s = 1 - evaluate.cos(qts, ris)
    D_i2t_u = 1 - evaluate.cos(qiu, rtu)
    D_t2i_u = 1 - evaluate.cos(qtu, riu)

    if args.sparse_label:
        Rel_s = S_s = evaluate.sim_mat(qls, rls, True)
        Rel_u = S_u = evaluate.sim_mat(qlu, rlu, True)
    else:
        Rel_s = qls.dot(rls.T)
        Rel_u = qlu.dot(rlu.T)
        S_s = (Rel_s > 0).astype(np.int32)
        S_u = (Rel_u > 0).astype(np.int32)

    # for _metr, _eval_fn, _S_s, _S_u in zip(
    #         model.metric_list, [evaluate.mAP, evaluate.nDCG], [S_s, Rel_s], [S_u, Rel_u]):
    for _metr, _eval_fn, _S_s, _S_u in zip(
            model.metric_list, [evaluate.mAP], [S_s], [S_u]):
        _i2t_s = _eval_fn(D_i2t_s, _S_s, args.mAP_at)
        _t2i_s = _eval_fn(D_t2i_s, _S_s, args.mAP_at)
        _i2t_u = _eval_fn(D_i2t_u, _S_u, args.mAP_at)
        _t2i_u = _eval_fn(D_t2i_u, _S_u, args.mAP_at)

        res[_metr+"_s_i2t"] = _i2t_s
        res[_metr+"_s_t2i"] = _t2i_s
        res[_metr+"_u_i2t"] = _i2t_u
        res[_metr+"_u_t2i"] = _t2i_u

    data =  {
        "qis": qis, "qts": qts, "qls": qls,
        "qiu": qiu, "qtu": qtu, "qlu": qlu,
        "ris": ris, "rts": rts, "rls": rls,
        "riu": riu, "rtu": rtu, "rlu": rlu,
    }

    return data, res


def train_lab_net(sess, model, writer, dataset, var):
    loader = dataset.iter_set(dataset.idx_train_s, args.batch_size,
            shuffle=True, image=False, text=False)
    # L_train = dataset.load_set(_idx_train,
    #         index=False, image=False, text=False, class_emb=False)
    avg_loss = MeanValue()
    for meta_index, label, cls_emb in loader:
        var["g_step_lb"] += 1

        # update pool
        # _emb, _reg_cls_emb = sess.run([model.emb_lb, model.reg_cls_emb_lb],
        #     feed_dict={model.class_emb: cls_emb, model.training: False})
        _emb = sess.run(model.emb_lb,
            feed_dict={model.class_emb: cls_emb, model.training: False})
        var["emb_lb"][meta_index] = _emb
        # var["cls_emb_lb"][meta_index] = _reg_cls_emb

        # S = calc_sim_mat(_emb, var["emb_lb"], label, L_train, sparse=args.sparse_label)
        S = var["S_bin"][meta_index]

        if args.sparse_label:
            label = one_hot(label, args.n_class)
        _, _loss, smy, smy_grad = sess.run(
            [model.train_lb, model.loss_lb, model.summary_lb, model.summary_grad_lb],
            feed_dict={model.class_emb: cls_emb,
                       model.class_emb_all: dataset.class_emb,  # class emb of both S & U
                       model.label: label,
                       model.sim_mat: S,
                       model.emb_pool: var["emb_lb"],
                    #    model.lr: var["lr_lb"],
                       model.training: True})

        if math.isnan(_loss):
            print("NaN: label net:", var["g_step_lb"])
            return None

        avg_loss.add(_loss)
        writer.add_summary(smy, var["g_step_lb"])
        writer.add_summary(smy_grad, var["g_step_lb"])
        # break

    model.record["loss_lb"].append(avg_loss.value()[0])
    return _loss


def train_dis_net(sess, model, writer, dataset, var):
    loader = dataset.iter_set(dataset.idx_train_s, args.batch_size,
            shuffle=True, index=False, label=False)
    avg_loss = MeanValue()
    for image, text, cls_emb in loader:
        var["g_step_adv"] += 1

        _emb_lb, _emb_im, _emb_tx = sess.run([model.emb_lb, model.emb_im, model.emb_tx],
            feed_dict={model.image: image, model.text: text, model.class_emb: cls_emb, model.training: False})

        _, _loss, smy, smy_grad = sess.run(
            [model.train_adv, model.loss_adv, model.summary_adv, model.summary_grad_adv],
            feed_dict={model.emb_lb: _emb_lb,
                       model.emb_im: _emb_im,
                       model.emb_tx: _emb_tx,
                    #    model.lr: var["lr_adv"],
                       model.training: True})

        if math.isnan(_loss):
            print("NaN: discriminator:", var["g_step_adv"])
            return None

        avg_loss.add(_loss)
        writer.add_summary(smy, var["g_step_adv"])
        writer.add_summary(smy_grad, var["g_step_adv"])
        # break

    model.record["loss_adv"].append(avg_loss.value()[0])
    return _loss


def train_img_net(sess, model, writer, dataset, var):
    loader = dataset.iter_set(dataset.idx_train_s, args.batch_size,
            shuffle=True, text=False, class_emb=False)
    # L_train = dataset.load_set(_idx_train,
    #         index=False, image=False, text=False, class_emb=False)
    avg_loss = MeanValue()
    for meta_index, label, image in loader:
        var["g_step_im"] += 1

        # _emb_batch = var["emb_lb"][meta_index]
        # S = calc_sim_mat(_emb_batch, var["emb_lb"], label, L_train, sparse=args.sparse_label)
        S = var["S_bin"][meta_index]

        _isfrom_im = sess.run(model.isfrom_im,
            feed_dict={model.image: image, model.training: False})

        if args.sparse_label:
            label = one_hot(label, args.n_class)
        _, _loss, smy, smy_grad = sess.run(
            [model.train_im, model.loss_im, model.summary_im, model.summary_grad_im],
            feed_dict={model.image: image,
                       model.label: label,
                       model.isfrom_im: _isfrom_im,
                       model.sim_mat: S,
                       model.emb_pool: var["emb_lb"],
                       model.cls_emb_pool: var["cls_emb_lb"],
                    #    model.lr: var["lr_im"],
                       model.training: True})

        if math.isnan(_loss):
            print("NaN: image net:", var["g_step_im"])
            return None

        avg_loss.add(_loss)
        writer.add_summary(smy, var["g_step_im"])
        writer.add_summary(smy_grad, var["g_step_im"])
        # break

    model.record["loss_im"].append(avg_loss.value()[0])
    return _loss


def train_txt_net(sess, model, writer, dataset, var):
    loader = dataset.iter_set(dataset.idx_train_s, args.batch_size,
            shuffle=True, image=False, class_emb=False)
    # L_train = dataset.load_set(_idx_train,
    #         index=False, image=False, text=False, class_emb=False)
    avg_loss = MeanValue()
    for meta_index, label, text in loader:
        var["g_step_tx"] += 1

        # _emb_batch = var["emb_lb"][meta_index]
        # S = calc_sim_mat(_emb_batch, var["emb_lb"], label, L_train, sparse=args.sparse_label)
        S = var["S_bin"][meta_index]

        _isfrom_tx = sess.run(model.isfrom_tx,
            feed_dict={model.text: text, model.training: False})

        if args.sparse_label:
            label = one_hot(label, args.n_class)
        _, _loss, smy, smy_grad = sess.run(
            [model.train_tx, model.loss_tx, model.summary_tx, model.summary_grad_tx],
            feed_dict={model.text: text,
                       model.label: label,
                       model.isfrom_tx: _isfrom_tx,
                       model.sim_mat: S,
                       model.emb_pool: var["emb_lb"],
                       model.cls_emb_pool: var["cls_emb_lb"],
                    #    model.lr: var["lr_tx"],
                       model.training: True})

        if math.isnan(_loss):
            print("NaN: text net:", var["g_step_tx"])
            return None

        avg_loss.add(_loss)
        writer.add_summary(smy, var["g_step_tx"])
        writer.add_summary(smy_grad, var["g_step_tx"])
        # break

    model.record["loss_tx"].append(avg_loss.value()[0])
    return _loss


def main(sess, model, writer, saver, dataset, logger):
    var = {}
    var['emb_lb'], var["cls_emb_lb"] = init_emb_pool(sess, model, dataset)
    L_train = dataset.load_set(dataset.idx_train_s,
        index=False, image=False, text=False, class_emb=False)
    var["L_train"] = L_train
    var["S_bin"] = evaluate.sim_mat(L_train, sparse=args.sparse_label)
    for k in ["lb", "adv", "im", "tx"]:
        var["g_step_{}".format(k)] = -1

    for epoch in range(args.epoch):
        logger("--- {} ---".format(epoch))

        print('- label net -')
        for idx in range(2):
            # _lr_lab_Up = lr_lab[epoch:]
            # var["lr_lb"] = _lr_lab_Up[idx]
            for train_labNet_k in range(k_lab_net // (idx + 1)):
                _loss = train_lab_net(sess, model, writer, dataset, var)
                if _loss is None:  # NaN
                    logger("NaN: label net: {} epoch, {} sub-epoch".format(epoch, train_labNet_k))
                    exit(1)
                # model.record['loss_lb'].append(_loss)
                logger('...epoch:\t{}, loss_labNet: {}'.format(epoch, _loss))
                if (train_labNet_k > 1) and (model.record['loss_lb'][-1] >= model.record['loss_lb'][-2]):
                    break
                # break  # train_labNet_k
            # break  # idx

        print('- discriminator -')
        for idx in range(2):
            # _lr_dis_Up = lr_dis[epoch:]
            # var["lr_adv"] = _lr_dis_Up[idx]
            for train_disNet_k in range(k_dis_net):
                _loss = train_dis_net(sess, model, writer, dataset, var)
                if _loss is None:  # NaN
                    logger("NaN: discriminator: {} epoch, {} sub-epoch".format(epoch, train_disNet_k ))
                    exit(1)
                # model.record['loss_adv'].append(_loss)
                logger('..epoch:\t{}, loss_adv: {}'.format(epoch, _loss))
                if (train_disNet_k > 1) and (model.record['loss_adv'][-1] <= model.record['loss_adv'][-2]):  # LESS-equal
                    break
                # break  # train_disNet_k
            # break  # idx

        print('- image net -')
        for idx in range(3):
            # _lr_img_Up = lr_img[epoch:]
            # var["lr_im"] = _lr_img_Up[idx]
            for train_imgNet_k in range(k_img_net // (idx + 1)):
                _loss = train_img_net(sess, model, writer, dataset, var)
                if _loss is None:  # NaN
                    logger("NaN: image net: {} epoch, {} sub-epoch".format(epoch, train_imgNet_k))
                    exit(1)
                if train_imgNet_k % 2 == 0:
                    # model.record['loss_im'].append(_loss)
                    logger('...epoch:\t{}, loss_imgNet: {}'.format(epoch, _loss))
                if (train_imgNet_k > 2) and (model.record['loss_im'][-1] >= model.record['loss_im'][-2]):
                    break
                # break  # train_imgNet_k
            # break  # idx

        print('- text net -')
        for idx in range(3):
            # _lr_txt_Up = lr_txt[epoch:]
            # var["lr_tx"] = _lr_txt_Up[idx]
            for train_txtNet_k in range(k_txt_net // (idx + 1)):
                _loss = train_txt_net(sess, model, writer, dataset, var)
                if _loss is None:  # NaN
                    logger("NaN: text net: {} epoch, {} sub-epoch".format(epoch, train_txtNet_k))
                    exit(1)
                if train_txtNet_k % 2 == 0:
                    # model.record['loss_tx'].append(_loss)
                    logger('...epoch:\t{}, Loss_txtNet: {}'.format(epoch, _loss))
                if train_txtNet_k > 2 and (model.record['loss_tx'][-1] >= model.record['loss_tx'][-2]):
                    break
                # break  # train_txtNet_k
            # break  # idx

        if epoch % args.test_per == 0:
            print("- test -")
            _data, _res = test(sess, model, dataset)
            for k in _res.keys():
                model.record[k].append(_res[k])

            for _metr in model.metric_list:
                for _su in "su":
                    _i2t, _t2i = _res[_metr+"_"+_su+"_i2t"], _res[_metr+"_"+_su+"_t2i"]
                    logger("{}_{}: epoch {}, i->t: {:.4f}, t->i: {:.4f}".format(
                        _metr, _su, epoch, _i2t, _t2i))

            logger.flush()

        # break  # epoch

    logger("--- final test ---")
    if not args.donot_save_model:
        saver.save(sess, osp.join(args.log_path, "TANSS"), global_step=args.epoch)
    _data, _res = test(sess, model, dataset)
    for _metr in model.metric_list:
        for _dir in ["i2t", "t2i"]:
            for _su in "su":
                _key = "{}_{}_{}".format(_metr, _su, _dir)
                model.record[_key].append(_res[_key])
                logger("{}: {:.4f}".format(_key, _res[_key]))

    return _data


if "__main__" == __name__:
    args.log_path = osp.join(args.log_path, args.dataset)
    # if osp.exists(args.log_path):
    #     shutil.rmtree(args.log_path)
    logger = Logger(args.log_path)
    for k, v in args._get_kwargs():
        logger("{}: {}".format(k, v))
    if 1 == args.tune:
        logger("zero-shot tuning mode")
    elif 2 == args.tune:
        logger("LRY's tuning mode")

    accumulated_metrics = collections.defaultdict(list)
    for i_run in range(args.n_run):
        logger("=== run: {} ===".format(i_run))
        logger("seed: {}".format(args.seed))
        run_path = osp.join(args.log_path, str(i_run))
        if osp.exists(run_path):
            shutil.rmtree(run_path)
        os.makedirs(run_path)

        if "wikipedia" == args.dataset:
            assert args.sparse_label and (not args.multi_label)
            dataset = wikipedia.Wikipedia("data/wikipedia", i_run, args.tune)
        elif "pascal-sentences" == args.dataset:
            assert args.sparse_label and (not args.multi_label)
            dataset = pascal_sentences.PascalSentences("data/pascal-sentences", args.tune)
        else:
            raise NotImplemented

        n_train = dataset.idx_train_s.shape[0]
        args.dim_image = dataset.images.shape[1]
        args.dim_text = dataset.texts.shape[1]
        args.n_class = len(np.unique(dataset.labels))
        args.dim_cls_emb = dataset.class_emb.shape[1]

        # reset seed
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(args.seed + 1)
        np.random.seed(args.seed + 2)
        tf.random.set_random_seed(args.seed + 3)

        gpu_conf = tf.ConfigProto()
        gpu_conf.gpu_options.allow_growth = True
        with tf.Session(config=gpu_conf) as sess:
            model = tanss.TANSS(n_train)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter(run_path)
            if args.resume_model:
                ckpt = tf.train.get_checkpoint_state(run_path)
                if ckpt and ckpt.model_checkpoint_path:
                    ckpt_name = osp.basename(ckpt.model_checkpoint_path)
                    saver.restore(sess, osp.join(run_path, ckpt_name))
                    logger("* success loading checkpoint: {}".format(ckpt_name))
                else:
                    logger("* checkpoint loading failed")

            _data = main(sess, model, writer, saver, dataset, logger)
            writer.close()

        sio.savemat(osp.join(run_path, "tanss.mat"), _data)
        for _metr in model.metric_list:
            for _su in "su":
                for _dir in ["i2t", "t2i"]:
                    _key = "{}_{}_{}".format(_metr, _su, _dir)
                    _v = model.record[_key][-1]
                    accumulated_metrics[_key].append(_v)

        print("- draw loss -")
        for k in model.record:
            if "loss_" not in k:
                continue
            v = model.record[k]
            if len(v) < 2:
                continue
            fig = plt.figure()
            plt.plot(v)
            plt.title(k)
            fig.savefig(osp.join(run_path, "{}.png".format(k)))
            plt.close(fig)

        print("- visual the results -")
        for _metr in model.metric_list:
            if len(model.record[_metr+"_s_i2t"]) < 2:
                continue
            fig = plt.figure()
            for _dir in ["i2t", "t2i"]:
                for _su in "su":
                    _key = "{}_{}_{}".format(_metr, _su, _dir)
                    plt.plot(model.record[_key], label=_key)
            plt.title(_metr)
            plt.legend()
            fig.savefig(osp.join(run_path, "{}.png".format(_metr)))
            plt.close(fig)

        tf.reset_default_graph()
        args.seed += 7  # different seed for different run
        # break  # run

    logger("------- training over -------")
    for _metr, _v_list in accumulated_metrics.items():
        _mean = np.mean(_v_list)
        _std = np.std(_v_list)
        # logger("{}: {:.4f} {} {:.4f}\n{}".format(_metr, _mean, chr(177), _std, _v_list))
        logger("{}: {:.4f} +- {:.4f}\n{}".format(_metr, _mean, _std, _v_list))
