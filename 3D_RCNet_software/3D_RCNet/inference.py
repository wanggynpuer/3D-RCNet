import torch
from DRCNet import HSIVit


def load_model(model_path):
    # 模型结构与训练时一致
    model = HSIVit(depths=[1, 2, 4, 2], dims=[32, 64, 128, 256], num_classes=16)  # 根据训练时的模型结构调整

    # 加载模型权重 state_dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # 加载到CPU

    # 去掉 "module." 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


if __name__ == '__main__':
    model = load_model('trained_model300.pkl')
    # batch_size = 32, channels = 1, depth = 200, height = 27, width = 27
    input_data = torch.randn(32, 1, 200, 27, 27)
    output = model(input_data)

    print("模型预测结果:", output)
