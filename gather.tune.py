import os
import os.path as osp
import glob
import re
import itertools
from tool import *


LOG_P = "log"
DATASET = ["wikipedia"]
MODE = "disjoint"
SPLIT = list(range(5))

"""
--- best ---
mAP_s: epoch 15, i2t 0.6274, t2i 0.9203
mAP_u: epoch 9, i2t 0.2729, t2i 0.2903
"""
best_pat = re.compile(
    r"\-+\sbest\s\-+\s+"
    r"mAP\_s\:\sepoch\s[0-9]+\,\si2t\s[0-9\.]+\,\st2i\s[0-9\.]+\s+"
    r"mAP\_u\:\sepoch\s[0-9]+\,\si2t\s[0-9\.]+\,\st2i\s[0-9\.]+"
    # r"nDCG\_s\:\sepoch\s[0-9]+\,\si2t\s[0-9\.]+\,\st2i\s[0-9\.]+\s+"
    # r"nDCG\_u\:\sepoch\s[0-9]+\,\si2t\s[0-9\.]+\,\st2i\s[0-9\.]+"
)
metric_list = ["mAP_s", "mAP_u"]#, "nDCG_s", "nDCG_u"]
metric_pat = [
    re.compile(r"mAP\_s\:\sepoch\s([0-9]+)\,\si2t\s([0-9\.]+)\,\st2i\s([0-9\.]+)"),
    re.compile(r"mAP\_u\:\sepoch\s([0-9]+)\,\si2t\s([0-9\.]+)\,\st2i\s([0-9\.]+)"),
    # re.compile(r"nDCG\_s\:\sepoch\s([0-9]+)\,\si2t\s([0-9\.]+)\,\st2i\s([0-9\.]+)"),
    # re.compile(r"nDCG\_u\:\sepoch\s([0-9]+)\,\si2t\s([0-9\.]+)\,\st2i\s([0-9\.]+)"),
]


def read_best(log_f):
    with open(log_f, "r") as f:
        log_txt = "".join(f.readlines())
        # print(log_txt)
    best_txt = best_pat.findall(log_txt)
    if len(best_txt) == 0:
        return None
    best_txt = best_txt[0]

    record = {}
    for _metr, _metr_pat in zip(metric_list, metric_pat):
        _res = _metr_pat.findall(best_txt)[0]
        _e = int(_res[0])
        _i2t = float(_res[1])
        _t2i = float(_res[2])
        _sum = _i2t + _t2i
        record[_metr] = [_e, _sum, _i2t, _t2i]

    return record


ALPHA = BETA = GAMMA = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
for dset in DATASET:
    print(dset)
    logger = Logger(LOG_P, "log.tune.{}.gamma.txt".format(dset))
    # for split_id in SPLIT:
    #     if 0 != split_id:
    #         continue
    #     logger("-- {}".format(split_id))
    # format: epoch, sum, i2t, t2i, param-value
    best = {_metr: [-1, 0, 0, 0, -1, -1, -1] for _metr in metric_list}
    # for _alpha, _beta, _gamma in itertools.product(
    #     [0.01, 0.1, 1], [0.1, 1, 10], [0.0001, 0.001, 0.01, 0.1, 1, 10],
    # ):
    for _gamma in GAMMA:
        log_f = glob.glob(osp.join(LOG_P, "tune_g{}".format(
            _gamma), dset, MODE, "log.*.txt"))
        # print(log_f)
        if 0 == len(log_f):
            logger("* NO LOG FILE: {}, {}".format(dset, _gamma))
            continue
        log_f = log_f[0]
        _res = read_best(log_f)
        if _res is None:
            logger("* read best failed: {}, {}".format(dset, _gamma))
            continue
        for _metr in metric_list:
            if _res[_metr][1] > best[_metr][1]:
                best[_metr][0] = _res[_metr][0]
                best[_metr][1] = _res[_metr][1]
                best[_metr][2] = _res[_metr][2]
                best[_metr][3] = _res[_metr][3]
                # best[_metr][4] = _alpha
                # best[_metr][5] = _beta
                best[_metr][4] = _gamma

        # break
    for _metr in metric_list:
        logger("{}: epoch: {}, sum: {}, i2t: {}, t2i: {}, gamma: {}".format(
            _metr, *best[_metr]))
    #     break
    # break
