from setuptools import setup, find_packages

setup(
    name='D_RCNet',
    version='0.2.7',
    packages=find_packages(where="."),
    package_dir={"D_RCNet": "D_RCNet"},  # 显式声明包目录
    install_requires=[
        'torch==1.12.0',  # PyTorch 版本
        'numpy>=1.23.5',  # NumPy 版本
        'gradio>=3.50.2', # GUI_app.py
        'fastapi>=0.115.12',  # 添加 FastAPI 依赖
        'uvicorn>=0.34.0',  # 添加 ASGI 服务器
        'python-multipart',  # 处理文件上传
    ],
    include_package_data=True,  # 包括数据文件

    package_data={  # 包含训练好的模型文件
        'D_RCNet': ['trained_model300.pkl'],
    },
    entry_points={
        'console_scripts': [
            'drcnet=D_RCNet.inference:main',
            'drcnet-api=D_RCNet.API_server:main'  # API入口
        ],
        'gui_scripts': [
            'drcnet-gui=D_RCNet.GUI_app:main'  # GUI入口
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    description="A package for 3D RCNet with pretrained model",
    author="wanggynpuer",
    author_email="3034711245@qq.com",
    url="https://github.com/wanggynpuer/3D-RCNet",  # GitHub 项目链接
)
