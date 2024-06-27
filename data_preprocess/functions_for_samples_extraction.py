import os
import h5py
import numpy as np


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


def h5_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]

    return data, label


def max_min_normalization(data):
    max_num = data.max()
    min_num = data.min()
    nl_data = (data - min_num) / (max_num - min_num)

    return nl_data


def padding(data, window_size):
    [m, n, c] = data.shape
    start_id = int(np.floor(window_size / 2))
    pad_data = np.zeros([m + window_size - 1, n + window_size - 1, c])
    pad_data[start_id: start_id + m, start_id: start_id + n] = data

    return pad_data


def samples_extraction(source_dir, save_dir, data_list_dir, window_size):
    HSI_data, HSI_gt = h5_loader(source_dir)
    HSI_data = max_min_normalization(HSI_data)
    s = window_size
    HSI_data = padding(HSI_data, s)

    [m, n] = HSI_gt.shape

    make_if_not_exist(save_dir)
    delete_if_exist(data_list_dir)

    for i in range(m):
        for j in range(n):
            if HSI_gt[i, j] > 0:
                label = HSI_gt[i, j]
                data = HSI_data[i:i + s, j:j + s, :].transpose([2, 0, 1])[np.newaxis]
                save_name = save_dir + 'samples_{}_{}.npy'.format(i + 1, j + 1)
                np.save(save_name, data)
                with open(data_list_dir, 'a') as f:
                    f.write(save_name + ' {}\n'.format(label))


def samples_division(list_dir, train_split_dir):
    samples_txt = open(list_dir).readlines()
    print(samples_txt)

    train_txt = open(train_split_dir).readlines()

    label_array = np.array([f.split(' ')[-1][:-1] for f in samples_txt], int)
    train_list = list_dir[: -4] + '_train.txt'
    test_list = list_dir[: -4] + '_test.txt'
    test_list_part = list_dir[: -4] + '_test_part.txt'
    delete_if_exist(train_list)
    delete_if_exist(test_list)
    delete_if_exist(test_list_part)

    for i in range(1, label_array.max() + 1):
        class_i_coord = np.where(label_array == i)
        samples_num_i = class_i_coord[0].size
        train_num_i = int(train_txt[i].split()[-1])
        kk = np.random.permutation(samples_num_i)
        if (train_num_i < samples_num_i):
            train_loc = class_i_coord[0][kk[:train_num_i]]
            test_loc = class_i_coord[0][kk[train_num_i:]]
            test_part_loc = class_i_coord[0][kk[train_num_i:train_num_i * 2]]
        else:
            train_num_i = samples_num_i / 2 + 1
            train_loc = class_i_coord[0][kk[:train_num_i]]
            test_loc = class_i_coord[0][kk[train_num_i:]]
            test_part_loc = class_i_coord[0][kk[train_num_i:train_num_i * 2]]

        with open(train_list, 'a') as f:
            for loc in train_loc:
                f.write(samples_txt[loc])
        with open(test_list, 'a') as f:
            for loc in test_loc:
                f.write(samples_txt[loc])
        with open(test_list_part, 'a') as f:
            for loc in test_part_loc:
                f.write(samples_txt[loc])


def samples_division_cv(list_dir, train_split_dir):
    samples_txt = open(list_dir).readlines()
    train_txt = open(train_split_dir).readlines()
    # val_txt = open(val_split_dir).readlines()

    label_array = np.array([f.split(' ')[-1][:-1] for f in samples_txt], int)
    train_list = list_dir[: -4] + '_train.txt'
    # val_list = list_dir[:-4]+'_val.txt'
    test_list = list_dir[: -4] + '_test.txt'

    delete_if_exist(train_list)
    delete_if_exist(test_list)
    # delete_if_exist(val_list)

    for i in range(1, label_array.max() + 1):
        class_i_coord = np.where(label_array == i)
        samples_num_i = class_i_coord[0].size
        train_num_i = int(train_txt[i].split()[-1])
        # val_num_i = int(val_txt[i].split()[-1])
        kk = np.random.permutation(samples_num_i)

        # train_loc = class_i_coord[0][kk[:train_num_i-val_num_i]]
        train_loc = class_i_coord[0][kk[:train_num_i]]
        # val_loc = class_i_coord[0][kk[train_num_i-val_num_i: train_num_i]]
        test_loc = class_i_coord[0][kk[train_num_i:]]

        with open(train_list, 'a') as f:
            for loc in train_loc:
                f.write(samples_txt[loc])
        with open(test_list, 'a') as f:
            for loc in test_loc:
                f.write(samples_txt[loc])
        # with open(val_list, 'a') as f:
        #     for loc in val_loc:
        #         f.write(samples_txt[loc])
