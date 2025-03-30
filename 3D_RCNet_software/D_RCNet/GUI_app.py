import gradio as gr
import numpy as np
import torch
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from D_RCNet.inference import predict


def visualize_3d(input_file):
    try:
        # 加载数据（增加空文件校验）
        if not input_file:
            raise ValueError("请上传有效的.npy文件")
        data = np.load(input_file.name)

        # 数据维度校验=
        if data.shape != (32, 1, 200, 27, 27):
            raise ValueError(f"数据维度需为 (32,1,200,27,27)，当前维度 {data.shape}")

        # 模型推理
        tensor_data = torch.tensor(data).float()
        if tensor_data.dim() != 5:
            tensor_data = tensor_data.unsqueeze(1)  # 自动补充缺失维度

        with torch.no_grad():
            prediction = predict(tensor_data)

        # 可视化
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sample_slice = data[0, 0, 100, :, :]
        x, y = np.meshgrid(np.arange(sample_slice.shape[1]), np.arange(sample_slice.shape[0]))
        ax.plot_surface(x, y, sample_slice, cmap='viridis')  # 改用surface提升渲染稳定性
        plt.savefig("result.png", dpi=96, bbox_inches='tight')  # 降低分辨率适配网页显示
        plt.close(fig)  # 显式释放资源

        return "result.png", f"预测类别：{np.argmax(prediction.numpy())}"

    except Exception as e:
        # 错误处理
        fig = plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, "ERROR",
                 ha='center', va='center',
                 fontdict={'size': 20, 'color': 'red'})
        plt.axis('off')
        plt.savefig("error.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return "error.png", f"错误原因：{str(e)}"


iface = gr.Interface(
    fn=visualize_3d,
    inputs=gr.File(label="上传3D数据(.npy)", file_types=[".npy"]),
    outputs=[
        gr.Image(label="3D渲染结果", type="filepath"),  # 强制指定文件路径类型
        gr.Textbox(label="分类预测")
    ],
    title="3D-RCNet可视化工具",
    examples=[["./sample.npy"] if os.path.exists("./sample.npy") else None],  # 动态校验示例文件
    allow_flagging="never"
)


def main():
    iface.launch(
        server_port=9000,
        show_error=True,  # 显示详细错误日志
        enable_queue=True  # 启用请求队列避免并发冲突
    )

if __name__ == "__main__":
    main()