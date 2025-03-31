import numpy as np
from pathlib import Path
import os


def create_sample_npy(output_path="sample.npy", mode="random"):
    target_shape = (32, 1, 200, 27, 27)

    # 数据生成逻辑
    if mode == "random":
        data = np.random.normal(loc=0, scale=1, size=target_shape).astype(np.float32)
    elif mode == "zeros":
        data = np.zeros(target_shape, dtype=np.float32)
    elif mode == "ones":
        data = np.ones(target_shape, dtype=np.float32)
    elif mode == "pattern":
        pattern = (np.indices((27, 27)).sum(axis=0) % 2).astype(np.float32)
        data = np.tile(pattern, (200, 1, 1))[np.newaxis, np.newaxis, ...]
        data = np.repeat(data, 32, axis=0).astype(np.float32)
    else:
        raise ValueError(f"无效模式: {mode}，支持模式：random/zeros/ones/pattern")

    # 显式验证
    if data.shape != target_shape:
        raise ValueError(f"形状错误：{data.shape} ≠ {target_shape}")
    if data.dtype != np.float32:
        raise TypeError(f"数据类型错误：{data.dtype}")

    # 保存文件
    np.save(output_path, data)
    output_abs_path = os.path.abspath(output_path)
    print(f"文件已生成：{output_abs_path}")
    return data

# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 生成随机数据样本（默认模式）
    create_sample_npy()

    # 生成其他类型数据示例：
    # create_sample_npy("zeros_sample.npy", mode="zeros")
    # create_sample_npy("pattern_sample.npy", mode="pattern")