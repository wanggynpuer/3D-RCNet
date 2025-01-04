import time
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch.optim as optim
import sys

sys.path.append('./models/')
from models.DRCNet import dict
from functions_for_training import *

sys.path.append('..')
from data_preprocess.functions_for_samples_extraction import h5_loader
from functions_for_evaluating import acc_calculation
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
import math
import get_cls_map

# Training settings
parser = argparse.ArgumentParser(description='HSI classification')
parser.add_argument('--dataset', type=str, default='Indian_pines')
parser.add_argument('--model_name', type=str, default='HSIVit')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--restore', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seed', default=0)
parser.add_argument('--model_save_interval', type=int, default=300)
parser.add_argument('--sheet_id', type=int, default=2)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--cutmix_prob', type=float, default=1)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--pin_mem', type=bool, default=True)
args = parser.parse_args()

train_data_txt = '../data_preprocess/data_list/{}_train.txt'.format(args.dataset)
# val_data_txt = '../data_preprocess/data_list/{}_val.txt'.format(args.dataset)
test_data_txt = '../data_preprocess/data_list/{}_test.txt'.format(args.dataset)
all_data_txt = '../data_preprocess/data_list/{}.txt'.format(args.dataset)

trained_model_dir = './train_info/' + args.dataset + '/' + args.model_name + '/'
train_info_record = trained_model_dir + args.model_name + '_{}'.format(args.sheet_id) + '.xls'

train_info_record_txt = trained_model_dir + args.model_name + '.txt'

torch.manual_seed(args.seed)

if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)  # set random seed

# data loaders
train_loader = torch.utils.data.DataLoader(
    data_loader(train_data_txt), num_workers=args.num_workers, pin_memory=args.pin_mem
    , batch_size=args.batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(
#     data_loader(val_data_txt), num_workers=args.num_workers, pin_memory=args.pin_mem
#     , batch_size=args.test_batch_size)

test_loader = torch.utils.data.DataLoader(
    data_loader(test_data_txt), num_workers=args.num_workers, pin_memory=args.pin_mem
    , batch_size=args.test_batch_size)

all_loader = torch.utils.data.DataLoader(
    data_loader(all_data_txt), num_workers=args.num_workers, pin_memory=args.pin_mem
    , batch_size=args.test_batch_size)
source_dir = '../data/HSI_datasets/data_h5/'
dataset_source_dir = source_dir + '{}.h5'.format(args.dataset)
HSI_data, HSI_gt = h5_loader(dataset_source_dir)
make_if_not_exist(trained_model_dir)

if args.dataset == 'PaviaU':
    num_cla = 9
elif args.dataset == 'Pavia':
    num_cla = 9
elif args.dataset == 'Indian_pines':
    num_cla = 16
else:
    num_cla = 15
    image_size = (1, 144, 27, 27)
    heads = 4
# model = DataParallel(dict[args.model_name](num_classes=num_cla, dropout_keep_prob=0))
# model = DataParallel(dict[args.model_name](num_classes=num_cla,depth=[2, 3, 3], dropout = [0., 0.],heads = [4, 8]))
model = DataParallel(dict[args.model_name](depths=[1, 2, 4, 2], dims=[32, 64, 128, 256], num_classes=num_cla))

if args.use_cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
T = args.epochs
t = args.epochs * 0.1
# cosine learning rate
lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.01 if 0.5 * (
            1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.01 else 0.5 * (
            1 + math.cos(math.pi * (epoch - t) / (T - t)))
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

start_epoch = 0
if args.restore and len(os.listdir(trained_model_dir)):
    model, start_epoch = model_restore(model, trained_model_dir)

# writer = SummaryWriter(logdir='log')


model_name = trained_model_dir + '/trained_model{}.pkl'.format(args.model_save_interval)
# torch.save(model.cpu().state_dict(), model_name)
# if args.use_cuda: model.cuda()
# train_loss, train_acc = val(model, train_loader, args)
# print('train_loss: {:.4f}, train_acc: {:.2f}%'.format(train_loss, train_acc))   
# val_loss, val_acc = val(model, val_loader, args)
# print('val_loss: {:.4f}, val_acc: {:.2f}%'.format(val_loss, val_acc))
#
test_loss, test_acc = val(model, test_loader, args)
print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(test_loss, test_acc))
# writer.add_scalars('loss',{'Train':train_loss,'Test':test_loss}, epoch)
# writer.add_scalars('acc',{'Train':train_acc,'Test':test_acc}, epoch)
# writer.add_scalars('lr',{'lr':lr},epoch)


# for epoch in range(start_epoch + 1, args.epochs + 1):
#     start = time.time()
#     lr = scheduler.get_last_lr()[0]
#     # train(epoch, lr, model, train_loader, optimizer, args)

#     end = time.time()
#     print('epoch: {} , cost {} seconds'.format(epoch, end - start))

#     if epoch % args.model_save_interval == 0:
#         model_name = trained_model_dir + '/trained_model{}.pkl'.format(epoch)
#         torch.save(model.cpu().state_dict(), model_name)
#         # if args.use_cuda: model.cuda()
#         # train_loss, train_acc = val(model, train_loader, args)
#         # print('train_loss: {:.4f}, train_acc: {:.2f}%'.format(train_loss, train_acc))   
#         # val_loss, val_acc = val(model, val_loader, args)
#         # print('val_loss: {:.4f}, val_acc: {:.2f}%'.format(val_loss, val_acc))
#         #
#         test_loss, test_acc = val(model, test_loader, args)
#         print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(test_loss, test_acc))
#         writer.add_scalars('loss',{'Train':train_loss,'Test':test_loss}, epoch)
#         writer.add_scalars('acc',{'Train':train_acc,'Test':test_acc}, epoch)
#         writer.add_scalars('lr',{'lr':lr},epoch)
#         with open(train_info_record_txt, 'a') as f:
#             f.write('timecost:{:.2f}, lr:{}, epoch:{}, train_loss:{:.4f}, train_acc:{:.2f},test_loss:{:.6f}, test_acc:{:.2f}'.format(
#                 (end-start)/60, optimizer.param_groups[0]['lr'], epoch, train_loss, train_acc, test_loss, test_acc) + '\n'
#             )
#         # with open(train_info_record_txt, 'a') as f:
#         #     f.write(
#         #         'timecost:{:.2f}, lr:{}, epoch:{}, train_loss:{:.4f}, train_acc:{:.2f}, val_loss:{:.6f}, val_acc:{:.2f}'.format(
#         #             (end - start) / 60, optimizer.param_groups[0]['lr'], epoch, train_loss, train_acc, val_loss,
#         #             val_acc) + '\n'
#         #         )
#     scheduler.step()

#     if args.use_cuda: model.cuda()
#     train_loss, train_acc = val(model, train_loader, args)
#     print('train_loss: {:.4f}, train_acc: {:.2f}%'.format(train_loss, train_acc))
#     # val_loss, val_acc = val(model, val_loader, args)
#     # print('val_loss: {:.4f}, val_acc: {:.2f}%'.format(val_loss, val_acc))

#     # writer.add_scalars('loss', {'Train': train_loss, 'Val': val_loss, }, epoch)
#     writer.add_scalars('loss', {'Train': train_loss}, epoch)
#     writer.add_scalars('acc', {'Train': train_acc}, epoch)
#     writer.add_scalars('lr', {'lr': lr}, epoch)

#     with open(train_info_record_txt, 'a') as f:
#         f.write(
#             'timecost:{:.2f}, lr:{}, epoch:{}, train_loss:{:.4f}, train_acc:{:.2f}'.format(
#                 (end - start) / 60, optimizer.param_groups[0]['lr'], epoch, train_loss, train_acc) + '\n'
#         )
#     # scheduler.step(val_loss)
# writer.close()

acc = acc_calculation(model, test_loader, args)
excel_write(train_info_record, acc)

source_dir = '../data/HSI_datasets/data_h5/'
dataset_source_dir = source_dir + '{}.h5'.format(args.dataset)
HSI_data, HSI_gt = h5_loader(dataset_source_dir)
get_cls_map.get_cls_map(model, all_loader, HSI_gt)
