# 3D-RCNet软件文档

1. **项目标题和简短描述**

   - **项目名称**：3D-RCNet
   - **简要说明**：用于高光谱图像分类的 3D 关系卷积网络模型

2. **功能概述**

   - **主要功能**：高光谱图像分类
   - **训练模型**：trained_model300.pkl
     - 300代表经过300轮迭代得到的模型
   - **高效性**：

3. **依赖项**

   - **Python 版本**：3.8及以上
   - **配置环境**
     -  `pip install -r requirements.txt`

4. **安装**

   - 命令行中输入以下命令
     - `pip install 3D_RCNet`

5. **使用示例**

   - 加载模型

     ``` model = load_model('trained_model300.pkl')``` 

     ```python
     def load_model(model_path):
         # 模型结构与训练时一致
         model = HSIVit(depths=[1, 2, 4, 2], dims=[32, 64, 128, 256], num_classes=16) 
         
         # 加载模型权重 state_dict
         state_dict = torch.load(model_path, map_location=torch.device('cpu'))
         
         # 去掉 "module." 前缀
         new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
     
         model.load_state_dict(new_state_dict)
         model.eval()
         return model
     ```

   - 生成数据（目前是随机生成）

     ```python
     # batch_size = 32, channels = 1, depth = 200, height = 27, width = 27
     input_data = torch.randn(32, 1, 200, 27, 27) 
     ```

   - 进行预测

     ``` output = model(input_data)``

   - 命令行进行

     - **随机输入**
       `python inference.py `
     - **指定输入输出**
       `python inference.py --input sample.npy --output prediction.npy`

6. **预训练模型**

   - 训练源代码：https://github.com/wanggynpuer/3D-RCNet

7. **许可证**

   ```
   								Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/
   ```

8. **致谢**

   论文出处：[3D-RCNet:Learning from Transformer to Build a 3D Relational ConvNet for Hyperspectral Image Classification](https://arxiv.org/abs/2408.13728)

   贡献者：**[wanggynpuer](https://github.com/wanggynpuer)**

   ​		**[respACz](https://github.com/respACz)**

   ​		[**taolijie11111**](https://github.com/taolijie11111)

   ​		[**zhouyang2002**](https://github.com/zhouyang2002)

### 调整和扩展：
- 待定