import scipy.io as sio
from os.path import join
import torch
from torch.utils.data import Dataset


class CUB2011(Dataset):
    def __init__(self, cnn_feats, cls_lbls, selected_clss):
        self.cnn_feats = cnn_feats
        self.cls_lbls = cls_lbls
        self.indices = []
        for cls in selected_clss:
            ls = [x[0]-1 for x in self.cls_lbls if x[1]==cls]
            self.indices += ls

    def __getitem__(self, idx):
        cnn_feat = torch.from_numpy(self.cnn_feats[self.indices[idx]])
        cls = self.cls_lbls[self.indices[idx]][1]-1
        return cnn_feat, cls

    def __len__(self):
        return len(self.indices)


def load_cub_mat_files(ds_path, cnn_det=True, easy_split=True):
    # mat file
    if cnn_det:
        cnn_feat_file = join(ds_path, "cnn_feat_7part_DET_ReLU.mat")
    else:
        cnn_feat_file = join(ds_path, "cnn_feat_7part_ATN_ReLU.mat")
    label_file = join(ds_path, "image_class_labels.mat")
    tfidf_file = join(ds_path, "11083D_TFIDF.mat")
    if easy_split:
        trainval_file = join(ds_path, "train_val_split_easy.mat")
        traintest_file = join(ds_path, "train_test_split_easy.mat")
    else:
        trainval_file = join(ds_path, "train_val_split_hard.mat")
        traintest_file = join(ds_path, "train_test_split_hard.mat")

    # Load CNN features
    cnn_feats = sio.loadmat(cnn_feat_file)
    cnn_feats = cnn_feats['cnn_feat'].transpose()
    # Load class labels
    cls_lbls = sio.loadmat(label_file)
    cls_lbls = cls_lbls['imageClassLabels']
    # Load TD-IDF features
    tf_idf = sio.loadmat(tfidf_file)
    tf_idf = tf_idf['PredicateMatrix']
    tf_idf = torch.from_numpy(tf_idf).type(torch.FloatTensor)
    # Load test and train indices
    trainval_cid = sio.loadmat(trainval_file)
    traintest_cid = sio.loadmat(traintest_file)

    return cnn_feats, cls_lbls, tf_idf, trainval_cid, traintest_cid


def get_cub2011(ds_path, val=False):
    num_cls = 200
    all_idx = list(range(1,num_cls+1))
    cnn_feats, cls_lbls, tf_idf, trainval_cid, traintest_cid = load_cub_mat_files(ds_path)
    train_clss = []
    test_clss = []
    val_clss = []
    if val:
        train_clss = trainval_cid['train_cid'][0].tolist()
        test_clss = trainval_cid['test_cid'][0].tolist()
        val_clss = [x for x in all_idx if not x in train_clss and not x in test_clss]
    else:
        train_clss = traintest_cid['train_cid'][0].tolist()
        test_clss = traintest_cid['test_cid'][0].tolist()
    #! -> indices start from 1!
    train_set = CUB2011(cnn_feats=cnn_feats, cls_lbls=cls_lbls, selected_clss=train_clss)
    val_set = CUB2011(cnn_feats=cnn_feats, cls_lbls=cls_lbls, selected_clss=val_clss)
    test_set = CUB2011(cnn_feats=cnn_feats, cls_lbls=cls_lbls, selected_clss=test_clss)

    return train_set, val_set, test_set, tf_idf
