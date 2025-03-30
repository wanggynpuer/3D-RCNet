import numpy as np
from pathlib import Path


def create_sample_npy(output_path="sample.npy", mode="random"):
    """
    生成符合 (32, 1, 200, 27, 27) 结构的有效.npy文件

    参数：
    - output_path: 输出文件路径（默认当前目录）
    - mode: 数据生成模式
      - random: 正态分布随机数（默认）
      - zeros: 全零测试数据
      - ones : 全一测试数据
      - pattern: 棋盘格测试模式
    """
    # 定义标准形状
    target_shape = (32, 1, 200, 27, 27)  # 模型输入的典型5D形状

    # 根据模式生成数据
    if mode == "random":
        data = np.random.normal(loc=0, scale=1, size=target_shape).astype(np.float32)
    elif mode == "zeros":
        data = np.zeros(target_shape, dtype=np.float32)
    elif mode == "ones":
        data = np.ones(target_shape, dtype=np.float32)
    elif mode == "pattern":
        # 生成棋盘格模式（验证空间维度）
        pattern = np.indices((27, 27)).sum(axis=0) % 2
        data = np.stack([pattern] * 200, axis=0)[np.newaxis, np.newaxis, ...]
        data = np.repeat(data, 32, axis=0).astype(np.float32)
    else:
        raise ValueError(f"无效模式: {mode}")

    # 保存文件
    np.save(output_path, data)
    print(f"文件已生成：{Path(output_path).resolve()}")

    # 验证数据
    assert data.shape == target_shape, f"形状错误：{data.shape} ≠ {target_shape}"
    assert data.dtype == np.float32, f"数据类型错误：{data.dtype}"
    return data


# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 生成随机数据样本（默认模式）
    create_sample_npy()

    # 生成其他类型数据示例：
    # create_sample_npy("zeros_sample.npy", mode="zeros")
    # create_sample_npy("pattern_sample.npy", mode="pattern")