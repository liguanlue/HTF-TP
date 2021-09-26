import cv2
import numpy as np
import pathlib
import pickle
from tqdm import tqdm
import random
labels_object = {'\"Pedestrian\"': 0,'\"Skater\"': 1,'\"Biker\"': 2,'\"Car\"': 3,'\"Cart\"': 3,'\"Bus\"': 3}

# DATASET_DIR = pathlib.Path('.').absolute().parent.parent.parent / 'datasets/sdd'
# SDD_RAW_DIR = DATASET_DIR / 'SDD'
# SDD_MAT_DIR = DATASET_DIR / 'SddMat'
SDD_RAW_DIR = pathlib.Path('/home/li/project/dagnet/datasets/sdd/SDD')

def get_annotation(data_dir, scene_dir):
    global labels_object
    print(data_dir)
    img = cv2.imread(str(scene_dir))
    with open(data_dir, 'r') as f_a:
        lines = []
        for line in f_a:
            line = line.strip().split(" ")
            line = [int(line[5]), int(line[0]), (float(line[2]) + float(line[4])) / 2.0,
                    (float(line[1]) + float(line[3])) / 2.0, labels_object[line[9]]]
            # frame object_id x y type
            if (line[2] < img.shape[0]) & (line[3] < img.shape[1]):
                lines.append(np.asarray(line))
            else:
                print("error data: {x}>{xmax} or {y}>{ymax}".format(x=line[2], xmax=img.shape[0], y=line[3],
                                                                 ymax=img.shape[1]))
    lines = np.stack(lines)

    return lines

def process_data(file_dir,setname):
    scene_dir = file_dir / 'label.jpg'
    data_a = get_annotation(file_dir/'annotations.txt', scene_dir)
    frame_list = np.unique(data_a[:, 0])
    print(frame_list.shape[0])
    mask = np.arange(0, frame_list.shape[0], 12)
    frame_list_sample = frame_list[mask]
    # 取出抽样的frame
    data_a_sample = None
    for frame in frame_list_sample:
        if data_a_sample is None:
            data_a_sample = data_a[data_a[:, 0] == frame, :]
        else:
            data_a_sample = np.concatenate((data_a_sample, data_a[data_a[:, 0] == frame, :]),axis=0)
    all_obj_sample = np.unique(data_a_sample[:, 1])
    all_data = None
    for obj in all_obj_sample:    # frame object_id x y type
        obj_allframe = data_a_sample[(data_a_sample[:, 1]==obj),:]
        if(obj_allframe.shape[0]<20):
            continue
        # index = random.randint(0, obj_allframe.shape[0]-20)
        index = 0
        if (all_data is None):
            all_data = obj_allframe[index:index+20,:]
        else:
            all_data = np.concatenate((all_data,obj_allframe[index:index+20,:]))
        # print(all_data.shape[0])
    path = '../trajnet_process/' +setname + '/stanford/' + str(file_dir.parents[0].name) + '_' + str(file_dir.name)[-1] + '.txt'
    np.savetxt(path,all_data,fmt='%.01f')

if __name__ == '__main__':

    train_set = {'bookstore': [0, 1, 2, 3], 'coupa': [3], 'deathCircle': [0, 1, 2, 3, 4], 'gates': [0,1,3,4,5,6,7,8],
                 'hyang': [4, 5, 6, 7, 9], 'little': [], 'nexus': [0, 1, 2, 3, 4, 7, 8, 9], 'quad': []}
    test_set = {'bookstore': [], 'coupa': [0, 1], 'deathCircle': [], 'gates': [2],
                'hyang': [0, 1, 3, 8], 'little': [0, 1, 2, 3], 'nexus': [5, 6], 'quad': [0, 1, 2, 3]}
    # validation_set = {'bookstore': [4, 5, 6], 'coupa': [2], 'deathCircle': [], 'gates': [0],
    #                   'hyang': [2, 10, 11, 12, 13, 14], 'little': [], 'nexus': [10, 11], 'quad': []}
    data_setsplit = [train_set, test_set]
    set_dir = {}
    set_name = {0: 'train', 1: 'test'}
    for i, setdir in enumerate(data_setsplit):
        for key, value in setdir.items():
            if value == []:
                continue
            for idx in value:
                video_num = 'video' + str(idx)
                set_dir.setdefault(set_name[i], []).append(SDD_RAW_DIR / key / video_num)

    print(set_dir)
    for setname, setdir in set_dir.items():
        for file_dir in tqdm(setdir, desc='Process ' + setname):
            print(file_dir)
            process_data(file_dir,setname)