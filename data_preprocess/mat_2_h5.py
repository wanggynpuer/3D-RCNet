import scipy.io as sio
import numpy as np
import h5py

# dataset = 'PaviaU'
# dataset_HSI = 'paviaU'
# dataset_gt = 'paviaU_gt'

# dataset = 'HoustonU'
# dataset_HSI = 'IGRSS_2013'
# dataset_gt = 'IGRSS_2013_gt'

# dataset = 'Indian_pines'
# dataset_HSI = 'indian_pines_corrected'
# dataset_gt = 'indian_pines_gt'

dataset = 'Pavia'
dataset_HSI = 'pavia'
dataset_gt = 'pavia_gt'

data_mat_dir = '../data/HSI_datasets/samples/'
data_h5_dir = '../data/HSI_datasets/data_h5/'

dataset_mat_dir = data_mat_dir + '{}/{}.mat'.format(dataset, dataset)
dataset_gt_dir = data_mat_dir + '{}/{}_gt.mat'.format(dataset, dataset)
dataset_h5_save_dir = data_h5_dir + '{}.h5'.format(dataset)

if dataset == 'HoustonU':
    matdata = h5py.File(dataset_mat_dir)
    HSI_data = np.transpose(matdata[dataset_HSI][:])
    print(HSI_data)

    matdatagr = h5py.File(dataset_gt_dir)
    HSI_gt = np.transpose(matdatagr[dataset_gt][:])
    print(HSI_gt)
else:
    a = sio.loadmat(dataset_mat_dir)
    # HSI_data = sio.loadmat(dataset_mat_dir)
    HSI_data = sio.loadmat(dataset_mat_dir)[dataset_HSI]
    #
    b = sio.loadmat(dataset_gt_dir)
    HSI_gt = sio.loadmat(dataset_gt_dir)[dataset_gt]

# -------------Houston--------------
# matdata = h5py.File(dataset_mat_dir)
# HSI_data = np.transpose(matdata[dataset_HSI][:])
# print(HSI_data)
#
# matdatagr = h5py.File(dataset_gt_dir)
# HSI_gt = np.transpose(matdatagr[dataset_gt][:])
# print(HSI_gt)
# ------------------IP、PU--------------------------
# a = sio.loadmat(dataset_mat_dir)
# HSI_data = sio.loadmat(dataset_mat_dir)
# HSI_data = sio.loadmat(dataset_mat_dir)[dataset_HSI]
#
# b = sio.loadmat(dataset_gt_dir)
# HSI_gt = sio.loadmat(dataset_gt_dir)[dataset_gt]

with h5py.File(dataset_h5_save_dir, 'w') as f:
    f['data'] = HSI_data
    f['label'] = HSI_gt
