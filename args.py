import argparse
# import numpy as np


parser = argparse.ArgumentParser(description='TANSS')
parser.add_argument('--log_path', type=str, default="log")
# parser.add_argument('--tune', action="store_true", default=False,
#                     help="add this flag to enable tuning mode")
parser.add_argument('--tune', type=int, default=0,
                    help="{0: non-tuning, 1: zero-shot mode, 2: LRY's mode}")
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--donot_save_model', action="store_true", default=False)
parser.add_argument('--resume_model', action="store_true", default=False)

parser.add_argument('--dataset', type=str, default="wikipedia")
parser.add_argument('--data_path', type=str, default="data/wikipedia")
parser.add_argument('--mix_mode', action="store_true", default=False)
parser.add_argument('--split_file', type=str,
                    default="data/wikipedia/split-0/split.wikipedia.0.mat")
parser.add_argument('--split_id', type=int, default=-1)
parser.add_argument('--sparse_label', action="store_true", default=False)
parser.add_argument('--multi_label', action="store_true", default=False)
parser.add_argument('--dim_image', type=int, default=4096)
parser.add_argument('--dim_text', type=int, default=300)
parser.add_argument('--n_class', type=int, default=10)
parser.add_argument('--dim_cls_emb', type=int, default=300)

parser.add_argument('--preprocess', action="store_true", default=False,
                    help="add this flag to proprocess the features")
parser.add_argument('--tanh_G', action="store_true", default=False,
                    help="add this flag to use tanh for activation in generator")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dim_emb', type=int, default=200, help="`K` in paper")
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--gamma', type=float, default=0.0001)

parser.add_argument('--epoch', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--mAP_at', type=int, default=-1)
args = parser.parse_args()


k_lab_net = 10
k_img_net = 15
k_txt_net = 15
k_dis_net = 1

# # Learning rate
# lr_lab = [np.power(0.1, x) for x in np.arange(2.0, 5 * args.epoch, 0.5)]
# lr_img = [np.power(0.1, x) for x in np.arange(4.5, 5 * args.epoch, 0.5)]
# lr_txt = [np.power(0.1, x) for x in np.arange(3.5, 5 * args.epoch, 0.5)]
# lr_dis = [np.power(0.1, x) for x in np.arange(3.0, 5 * args.epoch, 0.5)]
