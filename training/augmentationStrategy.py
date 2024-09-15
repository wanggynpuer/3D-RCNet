import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.autograd import Variable

def rand_bbox(size, lam):
    S = size[2]
    H = size[3]
    W = size[4]

    cut_rat = np.sqrt(1. - lam)
    cut_s = np.int(S * cut_rat)
    cut_h = np.int(H * cut_rat)
    cut_w = np.int(W * cut_rat)

    # uniform
    cs = np.random.randint(S)
    ch = np.random.randint(S)
    cw = np.random.randint(S)

    s1 = np.clip(cs - cut_s // 2, 0, S)
    s2 = np.clip(cs + cut_s // 2, 0, S)

    h1 = np.clip(ch - cut_h // 2, 0, H)
    h2 = np.clip(ch + cut_h // 2, 0, H)

    w1 = np.clip(cw - cut_w // 2, 0, W)
    w2 = np.clip(cw + cut_w // 2, 0, W)

    return s1, s2, h1, h2, w1, w2

'''
def cutmix(args,data,target,model):
    if args.beta > 0:
        # generate mixed sample
        lam = np.random.beta(args.beta, args.beta)
        rand_index = torch.randperm(data.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        s1, s2 = rand_bbox(data.size(), lam)
        data[:, :, :, :, s1:s2] = data[rand_index, :, :, :, s1:s2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((s2 - s1) / (data.size()[2]))
        # compute output
        output = model(data)
        loss = F.nll_loss(output, target_a) * lam + F.nll_loss(output, target_b) * (1. - lam)
    else:
        # compute output
        output = model(data)
        loss = F.nll_loss(output, target)
    return loss


class rotation_left90(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape1 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
        self.reshape2 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
    def forward(self, data, target, model):
        data = self.reshape1(data)
        data2 = data.copy()
        data2 = data.transpose(0,1,3,2,4)
        data2 = data2[:,:,::-1]
        data2 = self.reshape2(data2)
        output = model(data2)
        loss = F.nll_loss(output, target)
        return loss

class rotation_right90(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape1 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
        self.reshape2 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
    def forward(self, data, target, model):
        data = self.reshape1(data)
        data2 = data.copy()
        data2 = data.transpose(0,1,3,2,4)
        data2 = data2[:,:,:,::-1]
        data2 = self.reshape2(data2)
        output = model(data2)
        loss = F.nll_loss(output, target)
        return loss
class rotation_180(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape1 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
        self.reshape2 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
    def forward(self, data, target, model):
        data = self.reshape1(data)
        data2 = data.copy()
        data2 = data2[:,:,::-1]
        data2 = data2[:,:,:,::-1]
        data2 = self.reshape2(data2)
        output = model(data2)
        loss = F.nll_loss(output, target)
        return loss

class flipup(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape1 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
        self.reshape2 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
    def forward(self, data, target, model):
        data = self.reshape1(data)
        data2 = data.copy()
        data2 = data2[:, :, ::-1]
        data2 = self.reshape2(data2)
        output = model(data2)
        loss = F.nll_loss(output, target)
        return loss

class flipleft(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape1 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
        self.reshape2 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
    def forward(self, data, target, model):
        data = self.reshape1(data)
        data2 = data.copy()
        data2 = data2[:, :, :, ::-1]
        data2 = self.reshape2(data2)
        output = model(data2)
        loss = F.nll_loss(output, target)
        return loss
'''


# class augmentation(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.reshape1 = nn.Sequential(
#             Rearrange('b c s w h -> b c h w s')
#         )
#         self.reshape2 = nn.Sequential(
#             Rearrange('b c s w h -> b c h w s')
#         )
#     def cutmix(self,args, data, target, model):
#         # criterion = nn.CrossEntropyLoss()
#         if args.beta > 0:
#             # generate mixed sample
#             lam = np.random.beta(args.beta, args.beta)
#             rand_index = torch.randperm(data.size()[0]).cuda()
#             target_a = target
#             target_b = target[rand_index]
#             data = self.reshape1(data)
#             s1, s2 = rand_bbox(data.size(), lam)
#             data[:, :, :, :, s1:s2] = data[rand_index, :, :, :, s1:s2]
#             # adjust lambda to exactly match pixel ratio
#             lam = 1 - ((s2 - s1) / (data.size()[4]))
#             data = self.reshape2(data)
#             output = model(data)
#             loss =F.nll_loss(output, target_a) * lam + F.nll_loss(output, target_b) * (1. - lam)
#         else:
#             output = model(data)
#             loss = F.nll_loss(output, target)
#         return loss
#
#     def cutmixx(self,args, data, target, model):
#         criterion = nn.CrossEntropyLoss()
#         if args.beta > 0:
#             lam = np.random.beta(args.beta, args.beta)  # 随机的lam
#             rand_index = torch.randperm(data.size()[0]).cuda()  # 在batch中随机抽取一个样本，记为i
#             target_a = target
#             target_b = target[rand_index]  # 获取该样本的标签
#             data = self.reshape1(data)
#             bbx1, bby1, bbx2, bby2 = rand_bboxx(data.size(), lam)  # 随机产生一个box的四个坐标
#             data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2,:]  # 将样本i中box的像素值填充该批数据中相同区域
#             # adjust lambda to exactly match pixel ratio
#             lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-2] * data.size()[-3]))  # 按照box面积计算lam
#             data = self.reshape2(data)
#             # compute output
#             output = model(data)
#             loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)  # 通过lam融合损失
#         else:
#             output = model(data)
#             loss = F.nll_loss(output, target)
#         return loss
#
#     def flipleft(self, data, target, model):
#         # criterion = nn.CrossEntropyLoss()
#         data = self.reshape1(data)
#
#         data2 = torch.flip(data, dims=[3])
#         # data2 = data2[:, :, :, ::-1]
#
#         data2 = self.reshape2(data2)
#         output = model(data2)
#         loss = F.nll_loss(output, target)
#         # loss = criterion(output, target)
#         return loss
#     def flipsp(self, data, target, model):
#         # criterion = nn.CrossEntropyLoss()
#         data = self.reshape1(data)
#
#         data2 = torch.flip(data, dims=[4])
#         # data2 = data2[:, :, :, ::-1]
#
#         data2 = self.reshape2(data2)
#         output = model(data2)
#         loss = F.nll_loss(output, target)
#         return loss
#     def flipup(self, data, target, model):
#         # criterion = nn.CrossEntropyLoss()
#         data = self.reshape1(data)
#
#         data2 = torch.flip(data, dims=[2])
#         # data2 = data[:, :, ::-1]
#
#         data2 = self.reshape2(data2)
#         output = model(data2)
#         loss = F.nll_loss(output, target)
#         # loss = criterion(output, target)
#         return loss
#
#     def rotation_180(self, data, target, model):
#         # criterion = nn.CrossEntropyLoss()
#         data = self.reshape1(data)
#
#         data2 = torch.flip(data, dims=[2,3])
#         # data2 = data[:, :, ::-1]
#         # data2 = data2[:, :, :, ::-1]
#
#         data2 = self.reshape2(data2)
#         output = model(data2)
#         loss = F.nll_loss(output, target)
#         # loss = criterion(output, target)
#         # loss = F.nll_loss(output, target)
#         return loss
#     def rotation_right90(self, data, target, model):
#         # criterion = nn.CrossEntropyLoss()
#         data = self.reshape1(data)
#
#
#         data2 = data.transpose(3, 2)
#         data2 = torch.flip(data2, dims=[3])
#         # data2 = data2[:, :, :, ::-1]
#
#         data2 = self.reshape2(data2)
#         output = model(data2)
#         loss = F.nll_loss(output, target)
#         # loss = criterion(output, target)
#         # loss = F.nll_loss(output, target)
#         return loss
#     def rotation_left90(self, data, target, model):
#         # criterion = nn.CrossEntropyLoss()
#         data = self.reshape1(data)
#
#
#         data2 = data.transpose(3, 2)
#         data2 = torch.flip(data2, dims=[2])
#         # data2 = data2[:, :, ::-1]
#
#         data2 = self.reshape2(data2)
#         output = model(data2)
#         loss = F.nll_loss(output, target)
#         # loss = criterion(output, target)
#         # loss = F.nll_loss(output, target)
#         return loss
#
#     def xrotation_180(self, data, target, model):
#         data = self.reshape1(data)
#         data = torch.flip(data, dims=[3])
#
#         data2 = torch.flip(data, dims=[2,3])
#         # data2 = data[:, :, ::-1]
#         # data2 = data2[:, :, :, ::-1]
#
#         data2 = self.reshape2(data2)
#         output = model(data2)
#         # loss = F.nll_loss(output, target)
#         return output
#     def xrotation_right90(self, data, target, model):
#         data = self.reshape1(data)
#
#         data = torch.flip(data, dims=[3])
#
#         data2 = data.transpose(3, 2)
#         data2 = torch.flip(data2, dims=[3])
#         # data2 = data2[:, :, :, ::-1]
#
#
#         data2 = self.reshape2(data2)
#         output = model(data2)
#         # loss = F.nll_loss(output, target)
#         return output
#     def xrotation_left90(self, data, target, model):
#         data = self.reshape1(data)
#         data = torch.flip(data, dims=[3])
#
#         data2 = data.transpose(3, 2)
#         data2 = torch.flip(data2, dims=[2])
#         # data2 = data2[:, :, ::-1]
#
#         data2 = self.reshape2(data2)
#         output = model(data2)
#         # loss = F.nll_loss(output, target)
#         return output
#
#     def forward(self,args, data, target, model):
#         # criterion = nn.CrossEntropyLoss()
#         r1 = np.random.rand(1)
#         if r1 > args.cutmix_prob:
#             output = model(data)
#             loss = F.nll_loss(output, target)
#             # loss = criterion(output, target)
#         else:
#             loss = self.cutmix(args, data, target, model)
#
#         # elif r < 0.2:
#         #     loss = self.cutmix(args, data, target, model)
#         # elif r < 0.4:
#         #     loss = self.flipup(data, target, model)
#         # elif r < 0.6:
#         #     loss = self.flipleft(data, target, model)
#         # # elif r < 0.3:
#         # #     loss = self.flipsp(data, target, model)
#         # else:
#         #     x = np.random.rand(1)
#         #     if x < 0.3:
#         #         loss = self.rotation_180(data, target, model)
#         #     elif x < 0.6:
#         #         loss = self.rotation_left90(data,target,model)
#         #     else:
#         #         loss = self.rotation_right90(data,target,model)
#         #     # loss = self.cutmixx(args, data, target, model)
#
#         return loss
class augmentation(nn.Module):
    def __init__(self):
        super().__init__()

    def cutout(self,args, data):
        lam = np.random.beta(args.beta, args.beta)
        s1, s2, h1, h2, w1, w2 = rand_bbox(data.size(), lam)
        data[:, :, s1:s2, h1:h2, w1:w2] = 0.
        # s = data.size()[4]
        #
        # S = np.random.randint(s)
        #
        # y1 = np.clip(S - self.length // 2, 0, s)
        # y2 = np.clip(S + self.length // 2, 0, s)
        #
        # data[:,:,:,:,y1:y2] = 0.
        return data



    def cutmix(self,args, data, target):
        # criterion = nn.CrossEntropyLoss()
        if args.beta > 0:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(data.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]

            s1, s2, h1, h2, w1, w2 = rand_bbox(data.size(), lam)
            data[:, :, s1:s2, h1:h2, w1:w2] = data[rand_index, :, s1:s2, h1:h2, w1:w2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - (((s2 - s1)*(h1 - h2)*(w1 - w2)) / ((data.size()[2])*(data.size()[3])*(data.size()[4])))

            target = (target_a * lam + target_b * (1. - lam)).long()

        return data,target


    def flipleft(self, data):
        # criterion = nn.CrossEntropyLoss()
        data = torch.flip(data, dims=[4])
        # data2 = data2[:, :, :, ::-1]
        return data

    def flipup(self, data):
        # criterion = nn.CrossEntropyLoss()
        data = torch.flip(data, dims=[3])
        return data

    def rotation_180(self, data):
        # criterion = nn.CrossEntropyLoss()
        data = torch.flip(data, dims=[3,4])
        # data2 = data[:, :, ::-1]
        # data2 = data2[:, :, :, ::-1]
        return data
    def rotation_right90(self, data):
        # criterion = nn.CrossEntropyLoss()
        data2 = data.transpose(3, 4)
        data = torch.flip(data2, dims=[4])
        # data2 = data2[:, :, :, ::-1]
        return data
    def rotation_left90(self, data):
        # criterion = nn.CrossEntropyLoss()
        data2 = data.transpose(3, 4)
        data = torch.flip(data2, dims=[3])

        return data


    def forward(self,args, data, target, model):
        criterion = nn.CrossEntropyLoss()
        r, r1, r2, r3, r4, r5, r6 = [np.random.rand(1) for i in range(7)]

        if r < args.cutmix_prob:
            data = self.cutout(args, data)
            if r1 < 0.8:
                data, target = self.cutmix(args, data, target)
            # elif r2 < 0.3:
            #     data = self.flipup(data)
            # elif r3 < 0.3:
            #     data = self.flipleft(data)
            # elif r4 < 0.3:
            #     data = self.rotation_180(data)
            # elif r5 < 0.3:
            #     data = self.rotation_left90(data)
            # elif r6 < 0.3:
            #     data = self.rotation_right90(data)


        output = model(data)
        loss = criterion(output, target)
        # loss = F.nll_loss(output, target)

        return loss



class augmentationTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape1 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )
        self.reshape2 = nn.Sequential(
            Rearrange('b c s w h -> b c h w s')
        )

    def flipleft_test(self, data,model):
        data = self.reshape1(data)

        data2 = torch.flip(data, dims=[3])
        # data2 = data2[:, :, :, ::-1]

        data2 = self.reshape2(data2)
        output = model(data2)
        return output

    def flipup_test(self, data,model):
        data = self.reshape1(data)

        data2 = torch.flip(data, dims=[2])
        # data2 = data[:, :, ::-1]

        data2 = self.reshape2(data2)
        output = model(data2)
        return output

    def rotation_180_test(self, data,model):
        data = self.reshape1(data)
        data2 = torch.flip(data, dims=[2, 3])
        # data2 = data[:, :, ::-1]
        # data2 = data2[:, :, :, ::-1]

        data2 = self.reshape2(data2)
        output = model(data2)
        return output

    def rotation_right90_test(self, data,model):
        data = self.reshape1(data)
        data2 = data.transpose(3, 2)
        data2 = torch.flip(data2, dims=[3])
        # data2 = data2[:, :, :, ::-1]

        data2 = self.reshape2(data2)
        output = model(data2)
        return output

    def rotation_left90_test(self, data,model):
        data = self.reshape1(data)
        data2 = data.transpose(3, 2)
        data2 = torch.flip(data2, dims=[2])
        # data2 = data2[:, :, ::-1]

        data2 = self.reshape2(data2)
        output = model(data2)
        return output



def augment(args, data, target, model):
    w = augmentation()
    output = w(args, data, target, model)
    return output


# def augmentTest(data):
#     w = augmentationTest()
#     data = w(data)
#     return data

def augmentTest(data,model):
    w = augmentationTest()
    # output1 = w.flipleft_test(data,model)
    # output2 = w.flipup_test(data,model)
    #output = torch.min(output2, output1)
    # output = (output2+output1) / 2
    output1 = w.rotation_180_test(data,model)
    output2 = w.rotation_left90_test(data, model)
    output3 = w.rotation_right90_test(data, model)
    output4 = model(data)
    output = torch.max(output2, output1)
    output = torch.max(output4,output)
    output = torch.max(output3,output)
    # output = (output4+output2+output3+output1) / 4
    return output