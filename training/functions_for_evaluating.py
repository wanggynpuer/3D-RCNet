import torch
import numpy as np
import torch.utils.data as data
from sklearn.metrics import cohen_kappa_score
from torch.autograd import Variable
from augmentationStrategy import augmentTest
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score


class data_loader(data.Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)

    def __getitem__(self, index):
        sample_path = self.list_txt[index].split(' ')
        data_path = sample_path[0]
        label = sample_path[1][:-1]

        data = np.load(data_path)
        label = int(label) - 1

        return torch.from_numpy(data).float(), label

    def __len__(self):
        return self.length


# def OA_AA_K_cal(pre_label, tar_label):
#     acc=[]
#     samples_num = len(tar_label)
#     category_num=tar_label.max()+1
#     for i in range(category_num):
#         loc_i = np.where(tar_label==i)
#         OA_i = np.array(pre_label[loc_i]==tar_label[loc_i], np.float32).sum()/len(loc_i[0])
#         acc.append(OA_i)
#
#     OA = np.array(pre_label==tar_label, np.float32).sum()/samples_num
#     AA = np.average(np.array(acc))
#     # c_matrix = confusion_matrix(tar_label, pre_label)
#     # K = (samples_num*c_matrix.diagonal().sum())/(samples_num*samples_num - np.dot(sum(c_matrix,0), sum(c_matrix,1)))
#     K = cohen_kappa_score(tar_label, pre_label)
#     acc.append(OA)
#     acc.append(AA)
#     acc.append(K)
#     return np.array(acc)

def OA_AA_K_cal(pre_label, tar_label):
    oa = accuracy_score(tar_label, pre_label)
    print(oa)
    acc = []
    samples_num = len(tar_label)
    category_num = tar_label.max() + 1
    for i in range(category_num):
        loc_i = np.where(tar_label == i)
        OA_i = np.array(pre_label[loc_i] == tar_label[loc_i], np.float32).sum() / len(loc_i[0])
        acc.append(OA_i)

    OA = np.array(pre_label == tar_label, np.float32).sum() / samples_num
    AA = np.average(np.array(acc))
    # c_matrix = confusion_matrix(tar_label, pre_label)
    # K = (samples_num*c_matrix.diagonal().sum())/(samples_num*samples_num - np.dot(sum(c_matrix,0), sum(c_matrix,1)))
    K = cohen_kappa_score(tar_label, pre_label)
    acc.append(OA)
    acc.append(AA)
    acc.append(K)
    return np.array(acc)


@torch.no_grad()
def acc_calculation(model, val_loader, args):
    model.eval()
    # model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform())
    pre_label = torch.IntTensor([])
    tar_label = torch.IntTensor([])

    for data, target in val_loader:
        if args.use_cuda: data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
        # output = augmentTest(data, model)
        output = model(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        pre_label = torch.cat((pre_label, pred.cpu().int()), 0)
        tar_label = torch.cat((tar_label, target.int()), 0)

    return OA_AA_K_cal(pre_label.numpy(), tar_label.numpy())
