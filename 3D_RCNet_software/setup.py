from setuptools import setup, find_packages

setup(
    name='3D_RCNet',
    version='0.1.0',
    packages=find_packages(where='3D_RCNet'),
    install_requires=[
        'torch==1.12.0',  # PyTorch 版本
        'numpy==1.21.2',  # NumPy 版本
    ],
    include_package_data=True,  # 包括数据文件
    package_data={  # 包含训练好的模型文件
        '3D_RCNet': ['trained_model300.pkl'],
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
