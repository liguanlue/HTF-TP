#from IPython import embed
import glob
import pandas as pd
import pickle
import os
import torch
from torch import nn
from torch.utils import data
import random
import numpy as np
import cv2
from tqdm import tqdm
import copy
import dgl

# For scene_ID, 0: Lane, 1: Sidewalk, 2: Lawn, 3: Obstacle, 4: Door
color = [[88, 88, 88], [255, 127, 38], [14, 209, 69], [235, 28, 36], [0, 0, 254]]
labels_object = {'\"Pedestrian\"': 0, '\"Skater\"': 1, '\"Biker\"': 2, '\"Car\"': 3, '\"Cart\"': 3, '\"Bus\"': 3}
# 采样点数  街道：500

'''for sanity check'''
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def naive_social(p1_key, p2_key, all_data_dict):
    if abs(p1_key - p2_key) < 4:
        return True
    else:
        return False


def find_min_time(t1, t2):
    '''given two time frame arrays, find then min dist (time)'''
    min_d = 9e4
    t1, t2 = t1[:8], t2[:8]

    for t in t2:
        if abs(t1[0] - t) < min_d:
            min_d = abs(t1[0] - t)

    for t in t1:
        if abs(t2[0] - t) < min_d:
            min_d = abs(t2[0] - t)

    return min_d


def find_min_dist(p1x, p1y, p2x, p2y):
    '''given two time frame arrays, find then min dist'''
    min_d = 9e4
    p1x, p1y = p1x[:8], p1y[:8]
    p2x, p2y = p2x[:8], p2y[:8]

    for i in range(len(p1x)):
        for j in range(len(p1x)):
            if ((p2x[i] - p1x[j]) ** 2 + (p2y[i] - p1y[j]) ** 2) ** 0.5 < min_d:
                min_d = ((p2x[i] - p1x[j]) ** 2 + (p2y[i] - p1y[j]) ** 2) ** 0.5

    return min_d

# now_type, human, curr_keys[now_type][0], curr_keys[human][i], all_data_dict, time_thresh, dist_tresh
def social_and_temporal_filter(now_type, human, p1_key, p2_key, all_data_dict, time_thresh=48, dist_tresh=100):
    p1_traj, p2_traj = np.array(all_data_dict[now_type][p1_key]), np.array(all_data_dict[human][p2_key])
    p1_time, p2_time = p1_traj[:, 1], p2_traj[:, 1]
    p1_x, p2_x = p1_traj[:, 2], p2_traj[:, 2]
    p1_y, p2_y = p1_traj[:, 3], p2_traj[:, 3]

    if find_min_time(p1_time, p2_time) > time_thresh:
        return False
    if find_min_dist(p1_x, p1_y, p2_x, p2_y) > dist_tresh:
        return False

    return True
 # related_list.append([human,len(current_batch[human]),curr_keys[i]])
def mark_human(mask, sim_list):
    for i in range(len(sim_list)):
        for j in range(len(sim_list)):
            if i==j:
                continue
            rel_str = sim_list[i][0] + '<->' + sim_list[j][0]
            mask.setdefault(rel_str,[]).append([sim_list[i][1],sim_list[j][1]])

# human len(current_batch[human])-1 labels_type[1] len(scene_batch) - 1
def mark_scene(mask, relation):
    for i in range(len(relation)):
        rel_str = relation[i][0] + '<->' + relation[i][2]
        mask.setdefault(rel_str,[]).append([relation[i][1], relation[i][3]])

def get_scene(scene_dir):
    print(scene_dir)
    sample_nums = 1000
    img = cv2.imread(str(scene_dir))
    img2 = img[:, :, [2, 1, 0]]
    # 颜色阈值
    global color
    data_scene = []
    onehot_scene = []
    labels_scene = np.ones((img.shape[0], img.shape[1]), dtype=int) * -1
    id_class = []
    single_scene_img = []
    label_onehot = np.eye(5)
    for id in range(0, 5):
        lower_bound = np.array(color[id])
        upper_bound = lower_bound
        mask = cv2.inRange(img2, lower_bound, upper_bound)
        # 生成卷积核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 形态学处理:开运算
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        # 连通域划分
        ret, labels, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=4)
        # plt.imshow(labels, cmap='gray')
        # plt.show()
        connectedComponents_num = stats.shape[0]
        # print(connectedComponents_num)
        number = -1
        h, w = labels.shape
        for i in range(1, connectedComponents_num):
            if stats[i, 4] >= 500:  # Connected domains with an area greater than 500
                number += 1
                temp_img = cv2.inRange(labels, i, i)
                # single_scene_img.append(temp_img)
                id_class.append(id)  # 【Scene id, id after merging path】
                where = temp_img // 255 * len(id_class)
                labels_scene = labels_scene + where
                num = 0
                border_img = cv2.Canny(temp_img, 0, 0)
                points = []
                for j in range(h):
                    for k in range(w):
                        if border_img[j, k] == 255:
                            num += 1
                            points.append([j, k])
                # print(num)
                choose = np.linspace(0, num - 1, sample_nums)
                choose = choose.astype(int)
                points_sampled = []
                for j in choose:
                    points_sampled.append(points[j])

                data_scene.append(points_sampled)
                onehot_scene.append(label_onehot[:, id])

    #     data_scene = np.array(data_scene)
    #     onehot_scene = np.array(onehot_scene)

    return labels_scene, np.array(id_class), data_scene
    # return labels_scene, data_scene


def expand_getlabel(x, w, labels_total):
    box = labels_total[x[0] - w:x[0] + w, x[1] - w:x[1] + w]
    #     print(box)
    mask = np.unique(box)
    mask = np.delete(mask, np.where(mask == -1))
    #     print(mask)
    tmp = []
    for v in mask:
        tmp.append(np.sum(box == v))
    if tmp == []:
        return expand_getlabel(x, w + 5, labels_total)
    ts = np.max(tmp)
    max_v = mask[np.argmax(tmp)]
    #     print(max_v)
    return max_v


def find_scene(xy, labels_scene):
    labels = []
    # print(xy)
    for i in range(8):

        # print(xy[i][3])
        # print(xy[i][2])
        location = [int(xy[i][3]), int(xy[i][2])]
        #         print(location)
        # print(location)
        # print(labels_scene.shape)
        if location[0] >= labels_scene.shape[0]:
            location[0] = labels_scene.shape[0] - 1
        if location[1] >= labels_scene.shape[1]:
            location[1] = labels_scene.shape[1] - 1
        # print(location)
        if labels_scene[location[0], location[1]] != -1:
            label = labels_scene[location[0], location[1]]
        else:
            label = expand_getlabel(location, 5, labels_scene)
        if label not in labels:
            labels.append(label)
    return labels

def collect_scene_data(set_name, dataset_type='image',  root_path = ',,', verbose = True):
    assert  set_name in ['train','val','test']
    rel_path = '/trajnet_{0}/{1}/stanford'.format(dataset_type, set_name)
    # part_file = '/{}.txt'.format('*' if scene == None else scene)
    full_labels_scene , full_id_class, full_data_scene = [], [], []
    with open(set_name, 'rb') as f:
        all_file = pickle.load(f)
    print(set_name)
    print(all_file)

    for file in tqdm(all_file):

        scene_name = file[len(root_path + rel_path) + 1:-6] + file[-5]
        print(scene_name)
        scene_file = file[:-4] + '_label.jpg'
        labels_scene, id_class, data_scene = get_scene(scene_file)   # [h,w] [scene_num,1000,2]
        full_labels_scene.append(labels_scene)
        full_id_class.append(id_class)
        full_data_scene.append(data_scene)
    return full_labels_scene , full_id_class ,full_data_scene


def collect_data(set_name, dataset_type='process', batch_size=512, time_thresh=48, dist_tresh=100, scene=None,
                 verbose=True, root_path="../"):
    assert set_name in ['train', 'val', 'test']

    '''Please specify the parent directory of the dataset. In our case data was stored in:
        root_path/trajnet_image/train/scene_name.txt
        root_path/trajnet_image/test/scene_name.txt
    '''
    scene_name = root_path + "/scenedata_train.pickle"
    with open(scene_name, 'rb') as f:
        scene = pickle.load(f)
    # full_labels_scene , full_id_class, full_single_scene_img ,full_data_scene, full_onehot_scene = scene
    full_labels_scene, full_id_class, full_data_scene = scene

    rel_path = '/trajnet_{0}/{1}/stanford'.format(dataset_type, set_name)
    # labels_object = {'\"Pedestrian\"': 0, '\"Skater\"': 1, '\"Biker\"': 2, '\"Car\"': 3, '\"Cart\"': 3, '\"Bus\"': 3}
    # 初始化 ---------------------------------------------
    type_human = {0:'ped',1:'skater',2:'biker',3:'car'}
    type_scene = {0: 'lane',1:'sidewalk',2: 'lawn',3: 'Obstacle',4:'Door'}

    full_dataset = []
    full_masks_human = []
    full_masks_scene = []
    full_scene_data = []
    full_scene_data_picture = []
    full_picture = []
    full_picture_dict = []

    mask_human_batch = {}
    mask_scene_batch = {}
    current_batch = {}
    current_scene = {}
    current_scene_picture = {}
    current_picture = {}
    current_picture_dict = {}
    scene_w,scene_h = 0,0
    # scene_batch = []
    labels_dict = {}

    current_size = 0
    social_id = 0
    # 初始化===============================================================
    # part_file = '/{}.txt'.format('*' if scene == None else scene)
    # 读取数据 ——————————————————————————————
    with open(set_name+'filename.txt', 'rb') as f:
        all_file = pickle.load(f)
    print(set_name)
    print(all_file)

    for i, file in enumerate(tqdm(all_file)):

        data = np.loadtxt(fname=file, delimiter=' ')
        scene_dir = file[:-4] + '_label.jpg'
        img = cv2.imread(str(scene_dir))
        img1 = img[:, :, [2, 1, 0]].transpose(1, 0, 2)
        dim = (1500,1400)
        img2 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
        scene_name = file[len(root_path + rel_path) :-6] + file[-5]
        print(scene_name)
        # scene_file = file[:-4] + '_label.jpg'
        # full_labels_scene, full_id_class, full_single_scene_img, full_data_scene, full_onehot_scene
        labels_scene, id_class, data_scene  = full_labels_scene[i], full_id_class[i], full_data_scene[i] # [h,w] [scene_num,1000,2]

        data_by_id = {}
        for human in type_human.values():
            data_by_id[human] = {}
        # labels_object = {'\"Pedestrian\"': 0, '\"Skater\"': 1, '\"Biker\"': 2, '\"Car\"': 3, '\"Cart\"': 3, '\"Bus\"': 3}
        for frame_id, person_id, x, y, type in data:
            if person_id not in data_by_id[type_human[type]].keys():
                data_by_id[type_human[type]][person_id] = []
            data_by_id[type_human[type]][person_id].append([person_id, frame_id, x, y])

        for human in type_human.values():
            for person_id in data_by_id[human].keys():
                obj_frame = data_by_id[human][person_id]
                if(len(obj_frame)!=20):
                    print(len(obj_frame))

        all_data_dict = copy.deepcopy(data_by_id)
        left = 0
        if verbose:
            for human in type_human.values():
                print("Total {}: {}".format(human,len(list(data_by_id[human].keys()))))
                left += len(list(data_by_id[human].keys()))

        while left > 0:
            related_list = []
            curr_keys = {}
            for human in type_human.values():
                curr_keys[human] = list(data_by_id[human].keys())
            if current_size < batch_size:
                pass
            else:
                # 添加------------------------------------
                full_dataset.append(current_batch.copy())
                # mask_human_batch = np.array(mask_human_batch)
                full_masks_human.append(mask_human_batch)
                # mask_scene_batch = np.array(mask_scene_batch)
                full_masks_scene.append(mask_scene_batch)
                full_scene_data.append((current_scene.copy()))
                for name in current_scene_picture.keys():
                    current_scene_picture[name] = np.array(current_scene_picture[name])
                    # current_scene_picture[name] = current_scene_picture[name][:,:scene_w,:scene_h]
                full_scene_data_picture.append((current_scene_picture.copy()))
                full_picture.append(current_picture.copy())
                full_picture_dict.append(current_picture_dict.copy())
                print('len(scene_batch = {}'.format(len(current_scene)))
                # end 添加-------------------------------------

                # 初始化————————————————————————
                current_size = 0
                social_id = 0

                mask_human_batch = {}
                mask_scene_batch = {}
                current_batch = {}
                current_scene = {}
                current_scene_picture = {}
                current_picture = {}
                current_picture_dict = {}
                labels_dict = {}
                # end 初始化----------------------------------------------------------------------------------

            for human in type_human.values():
                now_type = human
                if len(curr_keys[human])>0:
                    current_batch.setdefault(human,[]).append((all_data_dict[human][curr_keys[human][0]]))
                    related_list.append([now_type, len(current_batch[human])-1, curr_keys[human][0]])  # -1
                    current_size += 1
                    if(scene_name not in current_picture.keys()):
                        current_picture[scene_name] = img2
                    current_picture_dict.setdefault(scene_name,{}).setdefault(human,[]).append(len(current_batch[human])-1)
                    del data_by_id[human][curr_keys[human][0]]  # 删除一个行人id
                    break
            for human in type_human.values():
                for i in range(1, len(curr_keys[human])):
                    if social_and_temporal_filter(now_type, human, curr_keys[now_type][0], curr_keys[human][i], all_data_dict, time_thresh, dist_tresh):
                        current_batch.setdefault(human,[]).append((all_data_dict[human][curr_keys[human][i]]))
                        related_list.append([human,len(current_batch[human])-1,curr_keys[human][i]])
                        current_size += 1
                        current_picture_dict.setdefault(scene_name, {}).setdefault(human, []).append(len(current_batch[human]) - 1)
                        del data_by_id[human][curr_keys[human][i]]
            mark_human(mask_human_batch, related_list)
            social_id += 1
            for item in related_list:
                labels = find_scene(all_data_dict[item[0]][item[2]], labels_scene)
                relation = []
                for label in labels:
                    name = scene_name + str(label)
                    if name not in labels_dict.keys():
                        current_scene.setdefault(type_scene[id_class[label]],[]).append(data_scene[label])
                        scene_big = np.zeros((2011,1980))
                        mask = np.where(labels_scene==label,1,0)
                        scene_big[:mask.shape[0],:mask.shape[1]] = mask
                        current_scene_picture.setdefault(type_scene[id_class[label]], []).append(scene_big.transpose(1, 0))
                        scene_w = max(scene_w, mask.shape[0])
                        scene_h = max(scene_h, mask.shape[1])
                        labels_dict[name] = len(current_scene[type_scene[id_class[label]]]) - 1
                    relation.append([item[0],item[1],type_scene[id_class[label]] ,labels_dict[name]]) # human len(current_batch[human])-1 labels_type[1] len(scene_batch) - 1
                mark_scene(mask_scene_batch, relation)
            left = 0
            for human in type_human.values():
                left += len(list(data_by_id[human].keys()))

    full_dataset.append(current_batch.copy())
    # mask_human_batch = np.array(mask_human_batch)
    full_masks_human.append(mask_human_batch)
    # mask_scene_batch = np.array(mask_scene_batch)
    full_masks_scene.append(mask_scene_batch)
    full_scene_data.append((current_scene.copy()))
    for name in current_scene_picture.keys():
        current_scene_picture[name] = np.array(current_scene_picture[name])
        # current_scene_picture[name] = current_scene_picture[name][:,:scene_w, :scene_h]
    full_scene_data_picture.append((current_scene_picture.copy()))
    full_picture.append(current_picture.copy())
    full_picture_dict.append(current_picture_dict.copy())

    return full_dataset, full_masks_human, full_masks_scene, full_scene_data, full_scene_data_picture, full_picture, full_picture_dict

def genetate_scene_data(train = True, test = False, root_path = '..'):
    if train:
        full_labels_scene, full_id_class, full_data_scene = collect_scene_data("train", root_path = root_path,verbose = True)
        scene = [ full_labels_scene , full_id_class, full_data_scene]
        scene_path = root_path + "/scenedata_train.pickle"
        with open(scene_path, 'wb') as f:
            pickle.dump(scene,f)
    if test:
        full_labels_scene, full_id_class, full_data_scene = collect_scene_data("test", root_path = root_path,verbose = True)
        scene = [ full_labels_scene , full_id_class, full_data_scene]
        scene_path = root_path + "/scenedata_test.pickle"
        with open(scene_path, 'wb') as f:
            pickle.dump(scene,f)


def generate_pooled_data(b_size, t_tresh, d_tresh, train=True, test=True, scene=None, verbose=True, root_path="..", dataset_type = 'process'):
    if train:
        full_dataset, full_masks_human, full_masks_scene, full_scene_data, full_scene_data_picture , full_picture, full_picture_dict= collect_data("train",
                                                                                                                  batch_size=b_size,
                                                                                                                  time_thresh=t_tresh,
                                                                                                                dist_tresh=d_tresh,
                                                                                                                  scene=scene,
                                                                                                                  verbose=verbose,
                                                                                                                root_path=root_path,
                                                                                                                  dataset_type = dataset_type)
        train = [full_dataset, full_masks_human, full_masks_scene, full_scene_data, full_scene_data_picture, full_picture, full_picture_dict]
        train_name = root_path + "/scene_pool_data_process/train_{0}_{1}_{2}_{3}.pickle".format(
            'all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh)
        with open(train_name, 'wb') as f:
            pickle.dump(train, f)

    if test:
        full_dataset, full_masks_human, full_masks_scene, full_scene_data, full_scene_data_picture, full_picture, full_picture_dict = collect_data("test",
                                                                                                                    batch_size=b_size,
                                                                                                                  time_thresh=t_tresh,
                                                                                                                dist_tresh=d_tresh,
                                                                                                                  scene=scene,
                                                                                                                verbose=verbose,
                                                                                                                  root_path = root_path,
                                                                                                                  dataset_type = dataset_type)
        test = [full_dataset, full_masks_human, full_masks_scene, full_scene_data, full_scene_data_picture, full_picture, full_picture_dict]
        test_name = root_path  + "/scene_pool_data_process/test_{0}_{1}_{2}_{3}.pickle".format(
            'all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh,
            d_tresh)  # + str(b_size) + "_" + str(t_tresh) + "_" + str(d_tresh) + ".pickle"
        with open(test_name, 'wb') as f:
            pickle.dump(test, f)

def initial_pos_get(traj_batches):
    batches = []
    for b in traj_batches:
        type_human = b.keys()
        starting_pos = {}
        for human in type_human:
            starting_pos[human] = b[human][:, 7, :].copy() / 1000  # starting pos is end of past, start of future. scaled down.
        batches.append(starting_pos)

    return batches


def calculate_loss(g, reconstructed_x, mean, log_var, criterion, interpolated_future, type_human):
    # reconstruction loss
    all_RCL_dest, all_ADL_traj, all_KLD = 0, 0, 0
    RCL_dest, ADL_traj, KLD = {}, {}, {}
    for human in type_human:
        RCL_dest[human] = criterion(g.nodes[human].data['dest'], reconstructed_x[human])
        all_RCL_dest += RCL_dest[human]
        ADL_traj[human] = criterion(g.nodes[human].data['future'], interpolated_future[human])  # better with l2 loss
        all_ADL_traj += ADL_traj[human]
    # kl divergence loss
        KLD[human] = -0.5 * torch.sum(1 + log_var[human] - mean[human].pow(2) - log_var[human].exp())
        all_KLD += KLD[human]

    return RCL_dest, KLD, ADL_traj, all_RCL_dest, all_ADL_traj, all_KLD


class SocialDataset(data.Dataset):

    def __init__(self, set_name="train", b_size=4096, t_tresh=60, d_tresh=50, data_scale = 1.86, past_length = 8, verbose=True, scene = None, root_path='..', device = 'cpu'):
        'Initialization'
        # root_path = 'tf/data/pecnet/social_pool_data'
        # root_path = '../social_pool_data'
        load_name = root_path  +'/scene_pool_data_process' "/{0}_{1}{2}_{3}_{4}.pickle".format(set_name,
                                                                           'all_' if scene is None else scene[:-2] +
                                                                                                        scene[-1] + '_',b_size, t_tresh, d_tresh)
        print(load_name)
        with open(load_name, 'rb') as f:
            data = pickle.load(f)

        traj, masks_human, masks_scene, scenes, scene_picture, full_picture, full_picture_dict = data
        # traj, masks = data

        traj_new, masks_human_new, masks_scene_new, scene_new, scene_new_picture, picture_new, picture_dict_new = [], [], [], [], [], [], []

        for i in range(len(traj)):
            t = traj[i]
            type_human = t.keys()
            reverse_t = {}
            for human in type_human:
                t[human] = np.array(t[human])
                t[human] = t[human][:,:,2:]
                reverse_t[human] = np.flip(t[human], axis=1).copy()
            traj_new.append(t)
            if set_name == 'train':
                traj_new.append(reverse_t)

            picture_new.append(full_picture[i])
            picture_dict_new.append(full_picture_dict[i])
            if set_name == 'train':
                picture_new.append(full_picture[i])
                picture_dict_new.append(full_picture_dict[i])

            masks_human_new.append(masks_human[i])
            if set_name == "train":
                # add second time for the reversed tracklets...
                masks_human_new.append(masks_human[i])

            masks_scene_new.append(masks_scene[i])
            if set_name == "train":
                # add second time for the reversed tracklets...
                masks_scene_new.append(masks_scene[i])

            scene_new.append(scenes[i])
            scene_new_picture.append(scene_picture[i])
            if set_name=='train':
                scene_new.append(scenes[i])
                scene_new_picture.append(scene_picture[i])

        self.mean = []
        for traj in traj_new:  # [batch, ]
            type_human = traj.keys()
            m_train = {}
            for human in type_human:
                m_train[human] = traj[human][:, :1, :].copy()  # 排查一下
                traj[human] -= traj[human][:, :1, :]
                traj[human] *= data_scale
            self.mean.append(m_train)

        initial_pos_batches = initial_pos_get(traj_new)
        self.trajectory_batches = []
        self.type_human = []
        self.type_scene = []

        for i, (traj, mask_human, mask_scene, initial_pos, scene_b, scene_picture) in enumerate(
                                                                                                        zip(traj_new.copy(),
                                                                                                            masks_human_new.copy(),
                                                                                                            masks_scene_new.copy(),
                                                                                                            initial_pos_batches.copy(),
                                                                                                            scene_new.copy(),
                                                                                                            scene_new_picture.copy())):

            type_human = traj.keys()
            type_scene = scene_b.keys()
            self.type_human.append(type_human)
            self.type_scene.append(type_scene)
            relation_dict = {}
            for human1 in type_human:
                for human2 in type_human:
                    rela_str = human1 + '<->' + human2
                    if (rela_str not in mask_human.keys()):
                        continue
                    relation_dict[(human1, 'adj', human2)] = mask_human[rela_str]
            for human in type_human:
                for scene in type_scene:
                    rela_str1 = human + '<->' + scene
                    if (rela_str1 not in mask_scene.keys()):
                        continue
                    relation_dict[(human, 'on', scene)] = mask_scene[rela_str1]
                    # rela_str2 = scene + '<->' + human
                    relation_dict[(scene, '_on', human)] = np.array(mask_scene[rela_str1])[:, [1, 0]].tolist()
            g = dgl.heterograph(relation_dict)
            g = g.to(device)
            for human in type_human:
                g.nodes[human].data['x'] = torch.DoubleTensor(traj[human][:, :past_length, :]).to(
                    device)
                delta = traj[human][:, :past_length, :]
                delta = np.diff(delta,axis=1)
                g.nodes[human].data['delta'] = torch.DoubleTensor(delta.copy()).to(device)
                g.nodes[human].data['y'] = torch.DoubleTensor(traj[human][:, past_length :, :]).to(
                    device)
                g.nodes[human].data['initial_pos'] = torch.DoubleTensor(initial_pos[human]).to(device)
                x_shape = g.nodes[human].data['x'].shape
                g.nodes[human].data['x'] = g.nodes[human].data['x'].contiguous().view(-1, x_shape[1] * x_shape[2]).to(device)
                g.nodes[human].data['delta'] = g.nodes[human].data['delta'].contiguous().view(-1, (x_shape[1]-1) * x_shape[2]).to(device)
                g.nodes[human].data['dest'] = g.nodes[human].data['y'][:, -1, :].to(device)
                y_shape = g.nodes[human].data['y'].shape
                g.nodes[human].data['future'] = g.nodes[human].data['y'][:, :-1, :].contiguous().view(y_shape[0],-1).to(device)

            # for scene in type_scene:
            # 	g.nodes[scene].data['data'] = torch.DoubleTensor(scene_b[scene]).contiguous().view(-1, 2000).to(device)
            # 	g.nodes[scene].data['picture'] = torch.DoubleTensor(scene_picture[scene]).to(device)
            # scene =  scene.contiguous().view(-1, scene.shape[1]*scene.shape[2])
            self.trajectory_batches.append(g)

        self.picture_batches = picture_new.copy()
        self.picture_dict_batches = picture_dict_new.copy()

        if verbose:
            print("Initialized social dataloader...")


"""
We've provided pickle files, but to generate new files for different datasets or thresholds, please use a command like so:
Parameter1: batchsize, Parameter2: time_thresh, Param3: dist_thresh
"""
if __name__ == '__main__':
    # dataset_type = 'image'
    dataset_type = 'process'
    set_name = ['train', 'test']
    root_path = ".."
    scene = None
    for i in range(2):
        rel_path = '/trajnet_{0}/{1}/stanford'.format(dataset_type, set_name[i])
        part_file = '/{}.txt'.format('*' if scene == None else scene)
        all_file = []
        for file in glob.glob(root_path + rel_path + part_file):
            all_file.append(file)

        with open('./' + set_name[i]+'filename.txt', 'wb') as f:
            pickle.dump(all_file, f)
        print(set_name[i])
        print(all_file)
    generate_pooled_data(200, 0, 100, train=True, test=True, verbose=True, root_path="..", dataset_type = dataset_type)
    # genetate_scene_data(train = False, test = True, root_path = '..')
# generate_pooled_data(512,0,25, train=True, verbose=True)
