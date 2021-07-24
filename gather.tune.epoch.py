import os
import os.path as osp
import glob
import re
import itertools
from tool import *


LOG_P = "log"
DATASET = ["wikipedia"]
MODE = "disjoint"
N_SPLIT = 5

"""
--- best ---
mAP_s: epoch 6, i2t 0.6245, t2i 0.8074
mAP_u: epoch 6, i2t 0.6245, t2i 0.8074
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
        # print("".join(log_txt))
    best_txt = best_pat.findall(log_txt)[0]

    record = {}
    for _metr, _metr_pat in zip(metric_list, metric_pat):
        _res = _metr_pat.findall(best_txt)[0]
        # print(_res)
        _e = int(_res[0])
        _i2t = float(_res[1])
        _t2i = float(_res[2])
        _sum = _i2t + _t2i
        record[_metr] = [_e, _sum, _i2t, _t2i]

    return record


for dset in DATASET:
    print(dset)
    logger = Logger(LOG_P, "log.tune.{}.epoch.txt".format(dset))
    for split_id in range(N_SPLIT):
        logger("\n- split id: {}".format(split_id))
        # format: beta[_metr][_param] = (epoch, sum, i2t, t2i)
        best = {_metr: -1 for _metr in metric_list}
        log_f = glob.glob(osp.join(
            LOG_P, "tune_e", dset, MODE, str(split_id), "log.*.txt"))
        # print(log_f)
        if 0 == len(log_f):
            logger("* NO LOG FILE: {}, {}, {}".format(dset, miss_rate, i_fold))
            continue
        log_f = log_f[0]
        _res = read_best(log_f)
        for _metr in metric_list:
            best[_metr] = _res[_metr][0]

        for _metr in metric_list:
            _e = best[_metr]
            logger("{}: epoch: {}".format(_metr, _e))
    #     break
    # break
