import torch
import argparse
import numpy as np
import hashlib
from D_RCNet.DRCNet import HSIVit
import os
from pathlib import Path

# 预定义模型文件的SHA-256哈希值
EXPECTED_HASH = "82ab8762421a8013863b19b4240ec10173b708a7eb60b6fe1b442d9aae8dca35"

# 获取当前模块所在目录
current_dir = Path(__file__).parent
model_path = current_dir / 'trained_model300.pkl'


def verify_model_hash(model_path):
    """验证模型文件完整性"""
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(65536):  # 64KB chunks
            sha256.update(chunk)
    file_hash = sha256.hexdigest()
    if file_hash != EXPECTED_HASH:
        raise ValueError(f"模型文件校验失败！预期哈希：{EXPECTED_HASH}，实际哈希：{file_hash}")


def load_model(model_path):
    """安全加载模型"""
    # 验证模型完整性
    verify_model_hash(model_path)

    # 初始化模型结构
    model = HSIVit(depths=[1, 2, 4, 2], dims=[32, 64, 128, 256], num_classes=16)

    try:
        # 安全加载state_dict
        state_dict = torch.load(
            model_path,
            map_location='cpu',
            # weights_only=True
        )

    except RuntimeError as e:
        if "Unsafe memory" in str(e):
            raise RuntimeError("检测到潜在危险数据，请使用可信模型文件") from e
        else:
            raise
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{str(e)}") from e

    # 适配多GPU训练保存的模型
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 安全加载参数
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def predict(input_data):
    """安全推理"""
    model = load_model(model_path)
    with torch.no_grad():
        output = model(input_data)
    return output


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='3D-RCNet Inference')
    parser.add_argument('--input', type=str, help='Input .npy data path')
    parser.add_argument('--output', type=str, default='result.npy', help='Output path')
    args = parser.parse_args()

    # 输入处理
    if args.input:
        data = np.load(args.input)
        tensor_data = torch.tensor(data).float().unsqueeze(1)  # 添加channel维度
    else:
        tensor_data = torch.randn(32, 1, 200, 27, 27)  # 生成测试数据

    try:
        result = predict(tensor_data)
        np.save(args.output, result.numpy())
        print(f"预测结果已保存至 {args.output}")
    except Exception as e:
        print(f"推理过程出错：{str(e)}")
        exit(1)


if __name__ == '__main__':
    main()