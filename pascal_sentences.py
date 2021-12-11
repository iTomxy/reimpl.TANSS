import os
import os.path as osp
import numpy as np
import scipy.io as sio
# from sklearn import preprocessing


class PascalSentences:
    def __init__(self, data_path, tune_mode=0):
        self.images = sio.loadmat(osp.join(
            data_path, "images.pascal-sentences.vgg19.4096d.mat"))["images"].astype(np.float32)  # [n, 4096]
        self.texts = sio.loadmat(osp.join(
            data_path, "texts.pascal-sentences.doc2vec.300.mat"))["texts"].astype(np.float32)  # [n, 300]
        self.labels = sio.loadmat(osp.join(
            data_path, "labels.pascal-sentences.mat"))["labels"][0].astype(np.int32)  # [n]
        self.class_emb = sio.loadmat(osp.join(
            data_path, "class_emb.pascal-sentences.Gnews-300d.mat"))["class_emb"].astype(np.float32)  # [c, 300]

        N_CLASS = len(np.unique(self.labels))
        assert 20 == N_CLASS

        print("images:", self.images.shape, self.images.min(), self.images.max())
        print("texts:", self.texts.shape, self.texts.min(), self.texts.max())
        print("labels:", self.labels.shape)
        print("class emb:", self.class_emb.shape)

        # class S/U division: 10 seen & 10 unseen
        class_set = np.arange(N_CLASS)
        seen_classes = np.random.choice(class_set, N_CLASS // 2, replace=False)
        self.unseen_classes = np.setdiff1d(class_set, seen_classes)
        seen_mask = np.zeros_like(self.labels, dtype=np.int8)  # belonging to seen classes
        for c in seen_classes:
            seen_mask[c == self.labels] = 1
        seen_mask = (seen_mask > 0)

        # data query/database splitting: 10/40 per class as query/database
        N_PC, N_Q_PC = 50, 10
        indices = np.arange(self.labels.shape[0])
        q_mask = np.zeros_like(self.labels, dtype=np.int8)  # belongging to query set
        for c in class_set:
            cls_mask = (c == self.labels)
            assert cls_mask.sum() == N_PC
            cls_idx = indices[cls_mask]
            q_idx = np.random.choice(cls_idx, N_Q_PC, replace=False)
            q_mask[q_idx] = 1
        q_mask = (q_mask > 0)

        self.idx_test_u = indices[(~ seen_mask) & q_mask]
        # self.idx_train_u = indices[(~ seen_mask) & (~ q_mask)]
        self.idx_ret_u = indices[(~ seen_mask) & (~ q_mask)]
        self.idx_test_s = indices[seen_mask & q_mask]
        self.idx_train_s = indices[seen_mask & (~ q_mask)]
        self.idx_ret_s = self.idx_train_s

        if 0 != tune_mode:
            if 1 == tune_mode:
                self._tune_mode()
            elif 2 == tune_mode:
                self._tune_mode_lry()
            else:
                raise NotImplemented

    def _tune_mode(self, u_rate=0.5, q_rate=0.3):
        Lsp_train = self.labels[self.idx_train_s]
        unique_c = np.unique(Lsp_train)
        # print("class set in train:", unique_c)
        nc_train = len(unique_c)
        u_set = set(unique_c[:int(u_rate * nc_train)].tolist())
        # print("new U set:", u_set)
        cls_id_set = {c: [] for c in unique_c}
        for _id, _c in zip(self.idx_train_s, Lsp_train):
            cls_id_set[_c].append(_id)

        _train_s, _train_u, _test_s, _test_u = [], [], [], []
        for _c in unique_c:
            _n = len(cls_id_set[_c])
            _nq = int(q_rate * _n)
            if _c in u_set:
                _test_u.extend(cls_id_set[_c][:_nq])
                _train_u.extend(cls_id_set[_c][_nq:])
            else:  # new seen class
                _test_s.extend(cls_id_set[_c][:_nq])
                _train_s.extend(cls_id_set[_c][_nq:])

        self.idx_test_s = np.asarray(_test_s)
        self.idx_train_s = np.asarray(_train_s)
        self.idx_ret_s = self.idx_train_s
        self.idx_test_u = np.asarray(_test_u)
        # self.idx_train_u = np.asarray(_train_u)
        self.idx_ret_u = np.asarray(_train_u)

    def _tune_mode_lry(self, q_rate=0.3):
        """LRY's tuning scheme"""
        n_train = self.idx_train_s.shape[0]
        n_val = int(q_rate * n_train)
        self.idx_test_s = self.idx_train_s[:n_val]
        self.idx_train_s = self.idx_train_s[n_val:]
        self.idx_ret_s = self.idx_train_s
        # UQ & UD are set to the same as SQ & SD
        # cuz the splitting is just like normal retrieval (or mix mode)
        self.idx_test_u = self.idx_test_s
        self.idx_ret_u = self.idx_ret_s

    def load_class_emb(self, class_id):
        """class_id: [n], 0-base class ID"""
        return self.class_emb[class_id]

    def iter_set(self, indices, batch_size=64, shuffle=False,
                 index=True, label=True, image=True, text=True, class_emb=True):
        """(meta index, label, image, text, class emb)"""
        assert int(index) + int(label) + int(image) + int(text) + int(class_emb) > 0, \
            "* nothing to load"
        meta_indices = np.arange(indices.shape[0])
        if shuffle:
            np.random.shuffle(meta_indices)
        for i in range(0, indices.shape[0], batch_size):
            res = []
            meta_idx = meta_indices[i: i + batch_size]
            idx = indices[meta_idx]
            label_batch = self.labels[idx]
            if index:
                res.append(meta_idx)
            if label:
                res.append(label_batch)
            if image:
                image_batch = self.images[idx]
                res.append(image_batch)
            if text:
                text_batch =  self.texts[idx]
                res.append(text_batch)
            if class_emb:
                cls_emb_batch = self.load_class_emb(label_batch)
                res.append(cls_emb_batch)

            if len(res) == 1:
                res = res[0]
            yield res

    def load_set(self, indices, shuffle=False,
                 index=True, label=True, image=True, text=True, class_emb=True):
        """(meta index, label, image, text, class emb)
        mainly used for loading training set labels
        """
        meta_indices = np.arange(indices.shape[0])
        if shuffle:
            np.random.shuffle(meta_indices)
        idx = indices[meta_indices]
        res = []
        lab = self.labels[idx]
        if index:
            res.append(meta_indices)
        if label:
            res.append(lab)
        if image:
            res.append(self.images[idx])
        if text:
            res.append(self.texts[idx])
        if class_emb:
            res.append(self.load_class_emb(lab))

        if len(res) == 1:
            res = res[0]
        return res


if "__main__" == __name__:
    pass
