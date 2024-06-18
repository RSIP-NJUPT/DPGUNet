import numpy as np
import time
from sklearn import preprocessing
import torch
from DPGUNet import DPGUNet
from utils import SegmentMap, get_samples_gt, gt_to_one_hot, draw_classification_map
from PIL import Image
import os
from config import OptInit
import scipy.io as sio

OPT = OptInit().get_args()

COUNT_PER_CLASS = np.zeros([OPT.num_classes], dtype=np.int64)
CORRECT_PER_CLASS = np.zeros([OPT.num_classes], dtype=np.int64)
PREDICT_PER_CLASS = np.zeros([OPT.num_classes], dtype=np.int64)
NUM_PER_CLASS = np.zeros([OPT.num_classes], dtype=np.int64)
TRAIN_TIME_ALL = 0.0
TEST_TIME_ALL = 0.0
IS_SAVE = False


def main():
    image_dir, gt_dir, segments_dir = OPT.image_dir, OPT.gt_dir, OPT.segments_dir
    img_list = os.listdir(image_dir)
    img_list.sort()
    gt_list = os.listdir(gt_dir)
    gt_list.sort()
    segments_list = os.listdir(segments_dir)
    segments_list.sort()
    num_trained_images = 0
    for i in range(len(img_list)):
        sar_img_path = os.path.join(image_dir, img_list[i])
        gt_path = os.path.join(gt_dir, gt_list[i])
        mat_path = os.path.join(segments_dir, segments_list[i])
        img = Image.open(gt_path)
        img = np.array(img)
        max_pixel_value = np.max(img)
        if max_pixel_value > 0: # 忽略全是背景的图像数据
            num_trained_images += 1
            train(sar_img_path, gt_path, mat_path, num_trained_images)

    # save log
    OPT.logger.info('=='*30)
    OPT.logger.info('\ndataset_name={}'.format(OPT.dataset_name))
    OPT.logger.info('train_ratio={}'.format(OPT.train_ratio))
    OPT.logger.info('val_ratio={}'.format(OPT.val_ratio))
    OPT.logger.info('learning rate={}'.format(OPT.lr))
    OPT.logger.info('num_classes={}'.format(OPT.num_classes))
    OPT.logger.info('epochs={}'.format(OPT.epochs))
    OPT.logger.info('num_images={}'.format(num_trained_images))


def train(sar_img_path=None, gt=None, mat=None, img_id=None):
    global IS_SAVE
    global TRAIN_TIME_ALL
    global TEST_TIME_ALL
    global COUNT_PER_CLASS
    global CORRECT_PER_CLASS
    global PREDICT_PER_CLASS
    global NUM_PER_CLASS

    OPT.logger.debug("{} {} {}".format("##"*20, img_id, "##"*20))
    OPT.logger.debug("image path  =  {}".format(sar_img_path))
    if sar_img_path.endswith(".mat"):
        data = sio.loadmat(sar_img_path)['data']
    else:
        data = Image.open(sar_img_path)
    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1], -1)
    gt = Image.open(gt)
    gt = np.array(gt)

    height, width, channels = data.shape
    data = np.reshape(data, [height * width, channels])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)  # 归一化 (std=1,mean=0)
    data = np.reshape(data, [height, width, channels])

    gt_reshape = np.reshape(gt, [-1])
    samples_count_list = []
    for i in range(OPT.num_classes):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        samples_count_list.append(samplesCount)
    OPT.logger.info(samples_count_list)

    train_samples_gt, test_samples_gt, val_samples_gt = get_samples_gt(OPT.seed, gt, OPT.num_classes,
                                                                       OPT.train_ratio, OPT.val_ratio)
    train_samples_gt_onehot = gt_to_one_hot(
        train_samples_gt, OPT.num_classes)
    test_samples_gt_onehot = gt_to_one_hot(
        test_samples_gt, OPT.num_classes)
    val_samples_gt_onehot = gt_to_one_hot(
        val_samples_gt, OPT.num_classes)
    # train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, OPT.num_classes]).astype(int)
    # test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, OPT.num_classes]).astype(int)
    # val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, OPT.num_classes]).astype(int)
    # Test_GT = np.reshape(test_samples_gt, [height, width])

    # 打印输出 train, val, test 的随机采样个数
    train_val_test_gt = [train_samples_gt,
                         val_samples_gt, test_samples_gt]
    for type in range(3):
        gt_reshape = np.reshape(train_val_test_gt[type], [-1])
        # OPT.logger.info("===============================")
        samples_count_list = []
        for i in range(OPT.num_classes):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            samples_count_list.append(samplesCount)
        OPT.logger.info(samples_count_list)
    """
    gt = [[0,1],        类别数 = 3
            [2,3]]      h=w=2
    GT_One_Hot = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>>train_samples_gt_onehot
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>>train_label_mask
        [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    """
    train_label_mask = np.zeros([height * width, OPT.num_classes])
    temp_ones = np.ones([OPT.num_classes])
    train_samples_gt = np.reshape(train_samples_gt, [height * width])
    for i in range(height * width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(
        train_label_mask, [height * width, OPT.num_classes])

    test_label_mask = np.zeros([height * width, OPT.num_classes])
    temp_ones = np.ones([OPT.num_classes])
    test_samples_gt = np.reshape(test_samples_gt, [height * width])
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(
        test_label_mask, [height * width, OPT.num_classes])

    val_label_mask = np.zeros([height * width, OPT.num_classes])
    temp_ones = np.ones([OPT.num_classes])
    val_samples_gt = np.reshape(val_samples_gt, [height * width])
    for i in range(height * width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(
        val_label_mask, [height * width, OPT.num_classes])

    tic = time.perf_counter()

    SM = SegmentMap(OPT.dataset_name, mat)
    association_matrices, adjacency_matrices = SM.getHierarchy()
    association_matrices = association_matrices[0:int(OPT.net_depth)]
    adjacency_matrices = adjacency_matrices[0:int(OPT.net_depth)]

    toc = time.perf_counter()
    OPT.logger.info('getHierarchy -- cost time:{}'.format(str(toc - tic)))
    OPT.logger.info("===============================")

    # data to device
    S_list_gpu = []
    A_list_gpu = []
    for i in range(len(association_matrices)):
        S_list_gpu.append(torch.from_numpy(
            np.array(association_matrices[i], dtype=np.float32)).to(OPT.device))
        # tmp = np.array(association_matrices[i], dtype=np.float32)
        # tmp2 = torch.from_numpy(tmp
        #     ).to(OPT.device)
        # S_list_gpu.append(tmp2)
        A_list_gpu.append(torch.from_numpy(
            np.array(adjacency_matrices[i], dtype=np.float32)).to(OPT.device))

    train_samples_gt = torch.from_numpy(
        train_samples_gt.astype(np.float32)).to(OPT.device)
    test_samples_gt = torch.from_numpy(
        test_samples_gt.astype(np.float32)).to(OPT.device)
    val_samples_gt = torch.from_numpy(
        val_samples_gt.astype(np.float32)).to(OPT.device)

    train_samples_gt_onehot = torch.from_numpy(
        train_samples_gt_onehot.astype(np.float32)).to(OPT.device)
    test_samples_gt_onehot = torch.from_numpy(
        test_samples_gt_onehot.astype(np.float32)).to(OPT.device)
    val_samples_gt_onehot = torch.from_numpy(
        val_samples_gt_onehot.astype(np.float32)).to(OPT.device)

    train_label_mask = torch.from_numpy(
        train_label_mask.astype(np.float32)).to(OPT.device)
    test_label_mask = torch.from_numpy(
        test_label_mask.astype(np.float32)).to(OPT.device)
    val_label_mask = torch.from_numpy(
        val_label_mask.astype(np.float32)).to(OPT.device)

    net_input = np.array(data, np.float32)
    net_input = torch.from_numpy(
        net_input.astype(np.float32)).to(OPT.device)

    net = DPGUNet(height, width, channels,
                  OPT.num_classes, S_list_gpu, A_list_gpu)

    net.to(OPT.device)
    if img_id > 1 and IS_SAVE:
        model_dir = os.path.join(OPT.work_dirs, 'model')
        net.load_state_dict(torch.load(
            os.path.join(model_dir, 'best_model.pt')))

    # loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
    def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
        """
            cross-entropy-loss:
            predict.shape=[h*w, c]= [21025, 16]
            reallabel_onehot.shape= [h*w, c]
            reallabel_mask.shape = [h*w, c]

            gt = [[0,1],        类别数 = 3
                    [2,3]]        h=w=2
            >>>train_samples_gt_onehot
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            >>>train_label_mask
                [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]]

        """
        real_labels = reallabel_onehot
        we = -torch.mul(real_labels, torch.log(predict + 1e-15))
        we = torch.mul(we, reallabel_mask)

        we2 = torch.sum(real_labels, 0)
        we2 = 1. / (we2 + 1)
        we2 = torch.unsqueeze(we2, 0)  # we2.shape=[1,c]
        we2 = we2.repeat([height * width, 1])  # we2.shape=[21025, c]
        we = torch.mul(we, we2)
        pool_cross_entropy = torch.sum(we)
        return pool_cross_entropy

    zeros = torch.zeros([height * width]).to(OPT.device).float()

    # evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
    def evaluate_performance(network_output,
                             train_samples_gt,
                             train_samples_gt_onehot,
                             require_AA_KPP=True):
        global COUNT_PER_CLASS
        global CORRECT_PER_CLASS
        global PREDICT_PER_CLASS
        global NUM_PER_CLASS
        if not require_AA_KPP:
            # 传入 train_samples_gt 算 train 的 OA, 传入 val_samples_gt 就算 val 的 OA...
            with torch.no_grad():
                available_label_idx = (
                    train_samples_gt != 0).float()
                available_label_count = available_label_idx.sum()  # 非背景像素总个数

                correct_prediction = torch.where(
                    torch.argmax(network_output, 1) == torch.argmax(
                        train_samples_gt_onehot, 1),
                    available_label_idx, zeros).sum()
                OA = correct_prediction / available_label_count

                return OA
        else:
            with torch.no_grad():
                # OA: Overall Accuracy
                available_label_idx = (
                    train_samples_gt != 0).float()
                available_label_count = available_label_idx.sum()  # 非背景像素总个数

                correct_prediction = torch.where(
                    torch.argmax(network_output, 1) == torch.argmax(
                        train_samples_gt_onehot, 1),
                    available_label_idx, zeros).sum()
                OA = correct_prediction / available_label_count
                OA = OA.cpu().numpy()

                # AA: Average Accuracy
                # zero_vector = np.zeros([OPT.num_classes])
                output_data = network_output.cpu().numpy()
                train_samples_gt = train_samples_gt.cpu().numpy()
                train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

                # output_data = np.reshape(output_data, [height * width, OPT.num_classes])
                idx = np.argmax(output_data, axis=-1) # 0~num_classes-1
                idx += 1 # 1~num_classes
                count_perclass = np.zeros(
                    [OPT.num_classes], dtype=np.int64)  # 统计每个类别的像素个数
                correct_perclass = np.zeros(
                    [OPT.num_classes], dtype=np.int64)  # 统计每个类别预测正确的像素个数
                predict_perclass = np.zeros(
                    [OPT.num_classes], dtype=np.int64)
                num_perclass = np.ones(
                    OPT.num_classes, dtype=np.int64)  # 统计该图中是否含有相应类别
                for x in range(len(train_samples_gt)):
                    if train_samples_gt[x] != 0:
                        count_perclass[int(
                            train_samples_gt[x] - 1)] += 1
                        if train_samples_gt[x] == idx[x]:
                            correct_perclass[int(
                                train_samples_gt[x] - 1)] += 1
                        predict_perclass[idx[x]-1] += 1

                COUNT_PER_CLASS += count_perclass
                CORRECT_PER_CLASS += correct_perclass
                PREDICT_PER_CLASS += predict_perclass
                # count_perclass = np.array(count_perclass, dtype=np.int64)
                num_perclass[count_perclass == 0] = 0
                NUM_PER_CLASS += num_perclass

                return OA

    optimizer = torch.optim.Adam(
        net.parameters(), lr=OPT.lr)  # weight_decay=0.0001

    # train the network
    best_loss = 99999
    best_OA = 0
    stop_flag = 0
    net.train()
    tic1 = time.perf_counter()
    with torch.autograd.set_detect_anomaly(True):
        for i in range(OPT.epochs + 1):
            optimizer.zero_grad()
            output = net(net_input)
            loss = compute_loss(
                output, train_samples_gt_onehot, train_label_mask)
            loss.backward(retain_graph=True)
            optimizer.step()  # Does the update
            if i % 10 == 0:
                with torch.no_grad():
                    net.eval()
                    output = net(net_input)
                    trainloss = compute_loss(
                        output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(
                        output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(
                        output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(
                        output, val_samples_gt, val_samples_gt_onehot)
                    OPT.logger.info("{}\ttrain loss={:.6f}\t train OA={:.6f} val loss={:.6f}\t val OA={:.6f}"
                                    .format(str(i + 1), trainloss, trainOA, valloss, valOA))

                    if valloss <best_loss or valOA> best_OA:
                        best_loss = valloss
                        best_OA = valOA
                        stop_flag = 0
                        model_dir = os.path.join(OPT.work_dirs, 'model')
                        os.makedirs(model_dir, exist_ok=True)
                        OPT.model_path = os.path.join(
                            model_dir, 'best_model.pt')
                        torch.save(net.state_dict(), OPT.model_path)
                        IS_SAVE = True
                        OPT.logger.info('save model...')

                net.train()
    toc1 = time.perf_counter()
    OPT.logger.info(
        "\n\n====================training done. starting evaluation...========================\n")
    training_time = toc1 - tic1
    TRAIN_TIME_ALL += training_time

    with torch.no_grad():
        net.load_state_dict(torch.load(OPT.model_path))
        net.eval()
        tic2 = time.perf_counter()
        output = net(net_input)
        toc2 = time.perf_counter()
        testloss = compute_loss(
            output, test_samples_gt_onehot, test_label_mask)
        testOA = evaluate_performance(
            output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True)
        OPT.logger.info("{}\ttest loss={:.6f}\t test OA={:.6f}".format(
            str(img_id), testloss, testOA))

        # save classification_map
        if OPT.save_map:
            pred = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
            sar_name, ext = os.path.splitext(os.path.basename(sar_img_path))
            pred_img_dir = os.path.join(
                OPT.work_dirs, OPT.dataset_name+'_classification_map',  sar_name+'.png')
            os.makedirs(pred_img_dir, exist_ok=True)
            draw_classification_map(pred, os.path.join(pred_img_dir, sar_name+'.png'),
                                    dataset_name=OPT.dataset_name)

        testing_time = toc2 - tic2
        TEST_TIME_ALL += testing_time


def calculate_performance():
    global COUNT_PER_CLASS
    global CORRECT_PER_CLASS
    global PREDICT_PER_CLASS
    global NUM_PER_CLASS
    global TRAIN_TIME_ALL
    global TEST_TIME_ALL
    OA = np.sum(CORRECT_PER_CLASS) / np.sum(COUNT_PER_CLASS)
    AC_PER_CLASS = CORRECT_PER_CLASS / COUNT_PER_CLASS
    AA = np.mean(AC_PER_CLASS)
    IOU_PER_CLASS = CORRECT_PER_CLASS / \
        (COUNT_PER_CLASS + PREDICT_PER_CLASS - CORRECT_PER_CLASS)
    MIOU = np.mean(IOU_PER_CLASS)
    P0 = np.sum(CORRECT_PER_CLASS) / np.sum(COUNT_PER_CLASS)
    Pe = np.sum(PREDICT_PER_CLASS * COUNT_PER_CLASS) / \
        (np.sum(COUNT_PER_CLASS) * np.sum(COUNT_PER_CLASS))
    KPP = (P0 - Pe) / (1 - Pe)

    # save information
    OPT.logger.info('OA={}'.format(OA))
    OPT.logger.info('NUM_PER_CLASS={}'.format(NUM_PER_CLASS))
    OPT.logger.info('AC_PER_CLASS={}'.format(AC_PER_CLASS))
    OPT.logger.info('AA={}'.format(AA))
    OPT.logger.info('IOU_PER_CLASS={}'.format(IOU_PER_CLASS))
    OPT.logger.info('MIOU={}'.format(MIOU))
    OPT.logger.info('KPP={}'.format(KPP))
    OPT.logger.info('TRAIN_TIME_ALL={}'.format(TRAIN_TIME_ALL))
    OPT.logger.info('TEST_TIME_ALL={}'.format(TEST_TIME_ALL))


if __name__ == '__main__':
    OPT.logger.info(os.getcwd())
    tic = time.perf_counter()
    main()
    calculate_performance()
    toc = time.perf_counter()
    all_time = toc - tic
    # save total run time
    OPT.logger.info('total run time={}'.format(all_time))
