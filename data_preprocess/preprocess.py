import argparse
from functions_for_samples_extraction import samples_extraction, samples_division, samples_division_cv

# Training settings
samples_dir = '../data/HSI_datasets/samples/'
source_dir = '../data/HSI_datasets/data_h5/'

parser = argparse.ArgumentParser(description='HSI classification')
parser.add_argument('--dataset', type=str, default='HoustonU')
parser.add_argument('--window_size', type=int, default=27)
args = parser.parse_args()

dataset_source_dir = source_dir + '{}.h5'.format(args.dataset)
samples_save_dir = samples_dir + '{}/'.format(args.dataset)
data_list_dir = './data_list/{}.txt'.format(args.dataset)
window_size = args.window_size
train_split_dir = './data_list/{}_split.txt'.format(args.dataset)
# val_split_dir = './data_list/{}_split_val.txt'.format(args.dataset)

samples_extraction(dataset_source_dir, samples_save_dir, data_list_dir, window_size)
# samples_division(data_list_dir, train_split_dir)
# samples_division_cv(data_list_dir, train_split_dir, val_split_dir)
samples_division_cv(data_list_dir, train_split_dir)
