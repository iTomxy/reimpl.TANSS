import math
import os
import os.path as osp
import random
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import tanss
import wikipedia
import evaluate
from wheel import *
from tool import *
from args import *


def init_emb_pool(sess, model, dataset):
    emb_pool, cls_emb_pool = [], []
    if args.mix_mode:
        _idx_train = dataset.idx_train
    else:
        _idx_train = dataset.idx_train_s
    loader = dataset.iter_set(_idx_train,
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
    if args.mix_mode:
        ql, qi, qt = gen_feature(sess, model, dataset, dataset.idx_test)
        rl, ri, rt = gen_feature(sess, model, dataset, dataset.idx_ret)

        if args.sparse_label:
            Rel = S = evaluate.sim_mat(ql, rl, True)
        else:
            Rel = ql.dot(rl.T).astype(np.int32)
            S = (Rel > 0).astype(np.int32)

        D_i2t = evaluate.euclidean(qi, rt)
        D_t2i = evaluate.euclidean(qt, ri)

        for _metr, _eval_fn, _sim_mat in zip(model.metric_list,
                [evaluate.mAP, evaluate.nDCG], [S, Rel]):
            res[_metr+"_i2t"] = _eval_fn(D_i2t, _sim_mat, args.mAP_at)
            res[_metr+"_t2i"] = _eval_fn(D_t2i, _sim_mat, args.mAP_at)

        data = {
            "ql": ql, "qi": qi, "qt": qt,
            "rl": rl, "ri": ri, "rt": rt,
        }
    else:
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

        for _metr, _eval_fn, _S_s, _S_u in zip(
                model.metric_list, [evaluate.mAP, evaluate.nDCG], [S_s, Rel_s], [S_u, Rel_u]):
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
    if args.mix_mode:
        _idx_train = dataset.idx_train
    else:
        _idx_train = dataset.idx_train_s
    loader = dataset.iter_set(_idx_train, args.batch_size,
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
    if args.mix_mode:
        _idx_train = dataset.idx_train
    else:
        _idx_train = dataset.idx_train_s
    loader = dataset.iter_set(_idx_train, args.batch_size,
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
    if args.mix_mode:
        _idx_train = dataset.idx_train
    else:
        _idx_train = dataset.idx_train_s
    loader = dataset.iter_set(_idx_train, args.batch_size,
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
    if args.mix_mode:
        _idx_train = dataset.idx_train
    else:
        _idx_train = dataset.idx_train_s
    loader = dataset.iter_set(_idx_train, args.batch_size,
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
    if args.mix_mode:
        _idx_train = dataset.idx_train
    else:
        _idx_train = dataset.idx_train_s
    L_train = dataset.load_set(_idx_train,
        index=False, image=False, text=False, class_emb=False)
    var["L_train"] = L_train
    var["S_bin"] = evaluate.sim_mat(L_train, sparse=args.sparse_label)
    for k in ["lb", "adv", "im", "tx"]:
        var["g_step_{}".format(k)] = -1

    for epoch in range(args.epoch):
        logger("--- {} ---".format(epoch))

        if epoch % 1 == 0:
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
                #     break  # train_labNet_k
                # break  # idx

        if epoch % 1 == 0:
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
                #     break  # train_disNet_k
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
            #     break  # train_imgNet_k
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
            #     break  # train_txtNet_k
            # break  # idx

        if args.tune:
            logger("- test -")
            _data, _res = test(sess, model, dataset)
            for k in _res.keys():
                model.record[k].append(_res[k])

            _to_save = False
            _main_metr = "nDCG" if args.multi_label else "mAP"
            if args.mix_mode:
                for _metr in model.metric_list:
                    _i2t, _t2i = _res[_metr+"_i2t"], _res[_metr+"_t2i"]
                    logger("{}: epoch {}, i->t: {:.4f}, t->i: {:.4f}".format(
                        _metr, epoch, _i2t, _t2i))

                    if (_i2t + _t2i) > (model.best[_metr][1] + model.best[_metr][2]):
                        model.best[_metr][0] = epoch
                        model.best[_metr][1] = _i2t
                        model.best[_metr][2] = _t2i
                        if _metr == _main_metr:
                            _to_save = True
            else:  # disjoint
                for _metr in model.metric_list:
                    for _su in "su":
                        _i2t, _t2i = _res[_metr+"_"+_su+"_i2t"], _res[_metr+"_"+_su+"_t2i"]
                        logger("{}_{}: epoch {}, i->t: {:.4f}, t->i: {:.4f}".format(
                            _metr, _su, epoch, _i2t, _t2i))

                        _key = "{}_{}".format(_metr, _su)
                        if (_i2t + _t2i) > (model.best[_key][1] + model.best[_key][2]):
                            model.best[_key][0] = epoch
                            model.best[_key][1] = _i2t
                            model.best[_key][2] = _t2i
                            if _metr == _main_metr:
                                _to_save = True

            if (not args.donot_save_model) and _to_save:
                saver.save(sess, osp.join(args.log_path, "TANSS"), global_step=epoch)

        # break  # epoch

    logger("--- final test ---")
    if not args.donot_save_model:
        saver.save(sess, osp.join(args.log_path, "TANSS"), global_step=args.epoch)
    _data, _res = test(sess, model, dataset)
    for _metr in model.metric_list:
        for _dir in ["i2t", "t2i"]:
            if args.mix_mode:
                _key = "{}_{}".format(_metr, _dir)
                model.record[_key].append(_res[_key])
                logger("{}: {:.4f}".format(_key, _res[_key]))
            else:  # disjoint
                for _su in "su":
                    _key = "{}_{}_{}".format(_metr, _su, _dir)
                    model.record[_key].append(_res[_key])
                    logger("{}: {:.4f}".format(_key, _res[_key]))

    _main_metr = "nDCG" if args.multi_label else "mAP"
    if args.mix_mode:
        _i2t, _t2i = _res[_main_metr+"_i2t"], _res[_main_metr+"_t2i"]
        _info_str = "{:.4f}it_{:.4f}ti".format(_i2t, _t2i)
    else:  # disjoint
        _i2t_s, _t2i_s = _res[_main_metr+"_s_i2t"], _res[_main_metr+"_s_t2i"]
        _i2t_u, _t2i_u = _res[_main_metr+"_u_i2t"], _res[_main_metr+"_u_t2i"]
        _info_str = "{:.4f}itu_{:.4f}tiu_{:.4f}its_{:.4f}tis".format(
            _i2t_u, _t2i_u, _i2t_s, _t2i_s)
    _f_name = osp.join(args.log_path, "tanss_{}.mat".format(_info_str))
    sio.savemat(_f_name, _data)

    print("- draw loss -")
    for k in model.record:
        if "loss_" not in k:
            continue
        fig = plt.figure()
        plt.plot(model.record[k])
        plt.title(k)
        fig.savefig(osp.join(args.log_path, "{}.png".format(k)))
        plt.close(fig)

    if args.tune:
        logger("--- best ---")
        for _metr in model.metric_list:
            if args.mix_mode:
                _e, _i2t, _t2i = model.best[_metr]
                logger("{}: epoch {}, i2t {:.4f}, t2i {:.4f}".format(
                    _metr, _e, _i2t, _t2i))
            else:  # disjoint
                for _su in "su":
                    _e, _i2t, _t2i = model.best["{}_{}".format(_metr, _su)]
                    logger("{}_{}: epoch {}, i2t {:.4f}, t2i {:.4f}".format(
                        _metr, _su, _e, _i2t, _t2i))

        print("- visual the results -")
        for _metr in model.metric_list:
            fig = plt.figure()
            for _dir in ["i2t", "t2i"]:
                if args.mix_mode:
                    _key = "{}_{}".format(_metr, _dir)
                    plt.plot(model.record[_key], label=_key)
                else:  # disjoint
                    for _su in "su":
                        _key = "{}_{}_{}".format(_metr, _su, _dir)
                        plt.plot(model.record[_key], label=_key)
            plt.title(_metr)
            plt.legend()
            fig.savefig(osp.join(args.log_path, "{}.png".format(_metr)))
            plt.close(fig)


if "__main__" == __name__:
    if args.tune:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_random_seed(args.seed)

    if args.multi_label:
        assert not args.sparse_label, "multi-label can NOT be sparse"
    if "wikipedia" == args.dataset:
        assert args.sparse_label and (not args.multi_label)
        print("* 2021.7.2: tuning with TEST set to repreduce results now ! fix it later")
        dataset = wikipedia.Wikipedia(
            args.data_path, args.split_file, args.mix_mode, args.tune, args.preprocess)
    else:
        raise NotImplemented
    if args.mix_mode:
        n_train = dataset.idx_train.shape[0]
    else:
        n_train = dataset.idx_train_s.shape[0]

    split_mode = "mix" if args.mix_mode else "disjoint"
    args.log_path = osp.join(args.log_path, args.dataset, split_mode)
    if args.split_id >= 0:
        args.log_path = osp.join(args.log_path, str(args.split_id))
    if args.preprocess:
        args.log_path = osp.join(args.log_path, "pre-proc")
    if osp.exists(args.log_path):
        os.system("rm -rf {}".format(args.log_path))

    logger = Logger(args.log_path)
    for k, v in args._get_kwargs():
        logger("{}: {}".format(k, v))
    if 1 == args.tune:
        logger("zero-shot tuning mode")
    elif 2 == args.tune:
        logger("LRY's tuning mode")

    gpu_conf = tf.ConfigProto()
    gpu_conf.gpu_options.allow_growth = True
    with tf.Session(config=gpu_conf) as sess:
        model = tanss.TANSS(n_train)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(args.log_path)
        if args.resume_model:
            ckpt = tf.train.get_checkpoint_state(args.log_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = osp.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, osp.join(args.log_path, ckpt_name))
                logger("* success loading checkpoint: {}".format(ckpt_name))
            else:
                logger("* checkpoint loading failed")
        main(sess, model, writer, saver, dataset, logger)
