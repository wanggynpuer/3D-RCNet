import torch
import time
import numpy as np
import torch.utils.data as data
from sklearn.metrics import cohen_kappa_score
from torch.autograd import Variable
from augmentationStrategy import augmentTest
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from torch.profiler import profile, record_function, ProfilerActivity
from thop import profile as thop_profile


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    if isinstance(model, torch.nn.DataParallel):    
        model = model.module
    model = model.to(device)

    # pre_label = torch.IntTensor([])
    # tar_label = torch.IntTensor([])
    pre_label = torch.empty(0, dtype=torch.int32, device=device)
    tar_label = torch.empty(0, dtype=torch.int32, device=device)


    total_inference_time = 0    # 推理时间统计
    batch_count = 0             # 记录批次数量
    flops, params = None, None  # 初始化 FLOPs 和参数量
    
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)

        if batch_idx == 0:
            dummy_input = data[:1]  
            dummy_input = dummy_input.to(device)
            flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)

        batch_count += 1
        start_time = time.time()     # 记录推理开始时间
        
        with torch.no_grad():
            output = model(data)
        end_time = time.time()       # 记录推理结束时间
        total_inference_time += (end_time - start_time) 

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        pre_label = torch.cat((pre_label, pred.int()), 0)
        tar_label = torch.cat((tar_label, target.int()), 0)

    avg_inference_time = total_inference_time / batch_count  # 平均推理时间
    metrics = {
        "flops": flops,
        "params": params,
        "total_inference_time": total_inference_time,
        "avg_inference_time": avg_inference_time,
    }
    print(f"FLOPs: {metrics['flops'] / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    print(f"Parameters: {metrics['params'] / 1e6:.2f} M")  # 转换为百万参数
    print(f"Total Inference Time: {metrics['total_inference_time']:.4f} seconds")
    print(f"Average Inference Time: {metrics['avg_inference_time']:.4f} seconds")

    return OA_AA_K_cal(pre_label.numpy(), tar_label.numpy())

@torch.no_grad()
def measure_model_performance(model, val_loader, device):
    model.eval()
    model = model.to(device)

    total_inference_time = 0    # 总推理时间
    batch_count = 0            # 批次数量
    dummy_input = None         # 用于统计 FLOPs 和参数量
    flops, params = None, None

    for batch_idx, (data, target) in enumerate(val_loader):
        data = data.to(device)

        if batch_idx == 0:
            dummy_input = data[:1]  # 选取一个样本
            flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)

        batch_count += 1
        start_time = time.time()  # 开始计时
        output = model(data)           # 模型推理
        end_time = time.time()    # 结束计时
        total_inference_time += (end_time - start_time)

    avg_inference_time = total_inference_time / batch_count  # 平均推理时间

    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    print(f"Parameters: {params / 1e6:.2f} M")  # 转换为百万参数
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"Average Inference Time per Batch: {avg_inference_time:.4f} seconds")

    return {
        "flops": flops,
        "params": params,
        "total_inference_time": total_inference_time,
        "avg_inference_time": avg_inference_time,
    }

