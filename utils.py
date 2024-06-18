import numpy as np
import random
import torch
from PIL import Image
import scipy.io as sio


class SegmentMap(object):
    def __init__(self, datasetname: np.array, mat_path:str=None):

        segs = sio.loadmat(mat_path)
        self.segs = segs['segmentmaps']

    def getHierarchy(self):
        """
        generate hierarchical node representations
        :param hierarchy: 从大到小的序列,表示每次池化的节点个数
        :return: association矩阵和邻接矩阵
        """
        segs = self.segs
        layers, h, w = self.segs.shape
        segs = np.concatenate([np.reshape([i for i in range(h * w)], [1, h, w]), segs], axis=0)
        layers = layers + 1
        association_matrices = []

        for i in range(layers - 1):
            M = np.zeros([np.max(segs[i]) + 1, np.max(segs[i + 1]) + 1], dtype=np.float32)
            l1 = np.reshape(segs[i], [-1]) # l1=[0,1,2...21024]
            l2 = np.reshape(segs[i + 1], [-1]) # l2=[0,1,2...2047]
            for x in range(h * w):
                if M[l1[x], l2[x]] != 1:
                    M[l1[x], l2[x]] = 1
            association_matrices.append(M)

        adjacency_matrices = []
        superpixelLabels = self.segs

        # 根据 segments 判定邻接矩阵
        for l in range(len(superpixelLabels)):
            segments = np.reshape(superpixelLabels[l], [h, w])
            superpixel_count = int(np.max(superpixelLabels[l])) + 1
            A = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)
            for i in range(h - 1):
                for j in range(w - 1):
                    sub = segments[i:i + 2, j:j + 2]
                    sub_max = np.max(sub).astype(np.int32)
                    sub_min = np.min(sub).astype(np.int32)

                    if sub_max != sub_min:
                        idx1 = sub_max
                        idx2 = sub_min
                        if A[idx1, idx2] != 0: continue
                        A[idx1, idx2] = A[idx2, idx1] = 1
            adjacency_matrices.append(A)

        return association_matrices, adjacency_matrices

def draw_classification_map(pred:torch.Tensor,
                            img_path: str,
                            dataset_name: str = 'munich'):
    if dataset_name == 'munich':
        color_map = {
            1: (222, 184, 135),
            2: (0, 100, 0),
            3: (203, 0, 0),
            4: (0, 0, 100),
        }
    else: # todo: add colormap of other datasets
        raise ValueError
    pred = np.array(pred).astype(np.uint8)
    rgb_array = np.zeros(
        (pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb_array[pred == class_id] = color
    rgb_image = Image.fromarray(rgb_array)
    rgb_image.save(img_path)


def gt_to_one_hot(gt, class_count):
    """
    Convet Gt to one-hot labels
    :param gt: 2D
    :param class_count:
    :return:
    """
    gt_one_hot = []  # 转化为one-hot形式的标签
    [height, width] = gt.shape
    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            gt_one_hot.append(temp)
    # gt_one_hot = np.reshape(gt_one_hot, [height, width, class_count])
    gt_one_hot = np.reshape(gt_one_hot, [-1, class_count]).astype(int)
    """
    gt = [[0,1],        类别数=3
            [2,3]]       h=w=2

    gt_one_hot = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>>gt_one_hot
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    gt_one_hot = np.reshape(gt_one_hot,[2,2,3])
    >>>gt_one_hot
    array([[[0, 0, 0],
            [1, 0, 0]],

            [[0, 1, 0],
            [0, 0, 1]]])
    """
    return gt_one_hot


def get_samples_gt(seed: int,
                    gt: np.array,
                      class_count: int,
                        train_ratio:float,
                          val_ratio:float):
    """
    类别 3, 0是背景
    gt = [[0,1,1],
          [2,2,2],
          [3,3,3]]

    return:
    train_samples_gt = [[0,0,0], #坐标(1,1)=2,(2,0)=3当做训练样本
                        [0,2,0],
                        [3,0,0]]

    val_samples_gt = [[0,0,0], #(1,0)=2当做验证样本
                      [2,0,0],
                      [0,0,0]]

    test_samples_gt = [[0,1,1], # 其余非背景像素全当做测试样本
                        [0,0,2],
                        [0,3,3]]

    """

    # 按照 train_ratio 和 val_ratio 随机划分样本为训练集和验证集，其余样本当做测试集
    random.seed(seed)
    [height, width] = gt.shape
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    val_rand_idx = []
    train_number_per_class = []
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]
        rand_idx = random.sample(rand_list,
                                    np.floor(samplesCount * train_ratio).astype('int32') +
                                    np.floor(samplesCount * val_ratio).astype('int32'))
        train_number_per_class.append(
            np.floor(samplesCount * train_ratio).astype('int32'))
        rand_real_idx_per_class = idx[rand_idx]
        train_rand_idx.append(rand_real_idx_per_class)
    train_rand_idx = np.array(
        train_rand_idx, dtype=object)  # list -> np.array
    train_data_index = []
    val_data_index = []
    for c in range(train_rand_idx.shape[0]):
        a = list(train_rand_idx[c])
        train_data_index = train_data_index + a[:train_number_per_class[c]]
        val_data_index = val_data_index + a[train_number_per_class[c]:]

    train_data_index = set(train_data_index)
    val_data_index = set(val_data_index)
    all_data_index = [i for i in range(len(gt_reshape))]
    all_data_index = set(all_data_index)

    test_data_index = all_data_index - train_data_index - val_data_index

    test_data_index = list(test_data_index)
    train_data_index = list(train_data_index)
    val_data_index = list(val_data_index)

    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass

    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])
    val_samples_gt = np.reshape(val_samples_gt, [height, width])

    return train_samples_gt, test_samples_gt, val_samples_gt


