# 基于pytorch实现的d2l第二版代码的个人批注以及**练习解答**
本仓库是基于pytorch实现的d2l第二版JupyterNotebook的个人批注以及练习解答，并尝试加入自己的一些理解或者拓展，欢迎一起交流学习！
## 安装d2l库
### CPU版本
1. [官网]((https://www.anaconda.com/))下载Anaconda;
2. 打开Anaconda Prompt，在命令界面输入：
> conda create --name d2l python=3.8 -y

`d2l`是环境名称，`3.8`是python版本，二者均可以随便改，推荐不改。

3. 激活创建的`d2l`环境，输入：

```shell
conda activate d2l
```

4. 安装需要的库，输入：

```shell
pip install d2l torch torchvison
```

### GPU版本
安装GPU版本，前3步不变，在第4步我们要安装torch的cuda版本。（`请自行搜索下载与NVIDIA显卡版本对应的CUDA和CUDNN`）

5. [pytorch官网](https://pytorch.org/)找到下载命令，在命令界面输入，如Win11+CUDA11.8：

> ```shell
> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

## d2l库补丁

由于d2l各个版本的torch实现都缺少部分函数与类的定义，本仓库以`1.0.3`版本的d2l库torch实现为基础进行修改与完善。

> 请将本仓库根目录下的`torch.py`文件复制到上一步创建的`d2l环境`中的`d2l库`中，替换掉库中的`torch.py。

`d2l环境`中的`d2l库`目录如下所示：
```bash
C:\Users\username\anaconda3\envs\d2l\Lib\site-packages\d2l
```
请将`username`修改为`你的用户名`，`第一个d2l`修改为你为`d2l环境`命名的`环境名称`。
## 目录

结构：
`视频序号: 代码章节`

- [03 安装](https://www.bilibili.com/video/BV18p4y1h7Dr/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_installation\index.ipynb](chapter_installation\index.ipynb)

- [04 数据操作+数据预处理](https://www.bilibili.com/video/BV1CV411Y7i4?p=1&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_preliminaries\ndarray.ipynb](chapter_preliminaries\ndarray.ipynb) + [chapter_preliminaries\pandas.ipynb](chapter_preliminaries\pandas.ipynb)

- [05 线性代数](https://www.bilibili.com/video/BV1eK4y1U7Qy/?spm_id_from=333.788.recommend_more_video.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_preliminaries\linear-algebra.ipynb](chapter_preliminaries\linear-algebra.ipynb)

- [06 矩阵计算](https://www.bilibili.com/video/BV1eZ4y1w7PY/?spm_id_from=333.788.recommend_more_video.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_preliminaries\calculus.ipynb](chapter_preliminaries\calculus.ipynb)

- [07 自动求导](https://www.bilibili.com/video/BV1KA411N7Px/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_preliminaries\autograd.ipynb](chapter_preliminaries\autograd.ipynb)

- [08 线性回归 + 基础优化算法](https://www.bilibili.com/video/BV1PX4y1g7KC/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_linear-networks\linear-regression.ipynb](chapter_linear-networks\linear-regression.ipynb) + [chapter_linear-networks\linear-regression-scratch.ipynb](chapter_linear-networks\linear-regression-scratch.ipynb) + [chapter_linear-networks\linear-regression-concise.ipynb](chapter_linear-networks\linear-regression-concise.ipynb)

- [09 Softmax 回归 + 损失函数 + 图片分类数据集](https://www.bilibili.com/video/BV1K64y1Q7wu/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_linear-networks\softmax-regression.ipynb](chapter_linear-networks\softmax-regression.ipynb) + [chapter_linear-networks\softmax-regression-scratch.ipynb](chapter_linear-networks\softmax-regression-scratch.ipynb) + [chapter_linear-networks\softmax-regression-concise.ipynb](chapter_linear-networks\softmax-regression-concise.ipynb) + [chapter_linear-networks\image-classification-dataset.ipynb](chapter_linear-networks\image-classification-dataset.ipynb)

- [10 多层感知机](https://www.bilibili.com/video/BV1hh411U7gn/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_multilayer-perceptrons\mlp.ipynb](chapter_multilayer-perceptrons\mlp.ipynb) + [chapter_multilayer-perceptrons\mlp-scratch.ipynb](chapter_multilayer-perceptrons\mlp-scratch.ipynb) + [chapter_multilayer-perceptrons\mlp-concise.ipynb](chapter_multilayer-perceptrons\mlp-concise.ipynb)

- [11 模型选择 + 过拟合和欠拟合](https://www.bilibili.com/video/BV1kX4y1g7jp/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_multilayer-perceptrons\underfit-overfit.ipynb](chapter_multilayer-perceptrons\underfit-overfit.ipynb)

- [12 权重衰退](https://www.bilibili.com/video/BV1UK4y1o7dy/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_multilayer-perceptrons\weight-decay.ipynb](chapter_multilayer-perceptrons\weight-decay.ipynb)

- [13 丢弃法](https://www.bilibili.com/video/BV1Y5411c7aY/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_multilayer-perceptrons\dropout.ipynb](chapter_multilayer-perceptrons\dropout.ipynb)

- [14 数值稳定性 + 模型初始化和激活函数](https://www.bilibili.com/video/BV1u64y1i75a/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_multilayer-perceptrons\numerical-stability-and-init.ipynb](chapter_multilayer-perceptrons\numerical-stability-and-init.ipynb)

- [15 实战:Kaggle房价预测 + 课程竞赛:加州2020年房价预测](https://www.bilibili.com/video/BV1NK4y1P7Tu/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_multilayer-perceptrons\kaggle-house-price.ipynb](chapter_multilayer-perceptrons\kaggle-house-price.ipynb)

- [前向传播、反向传播、计算图：chapter_multilayer-perceptrons\backprop.ipynb](chapter_multilayer-perceptrons\backprop.ipynb)

- [16 Pytorch 神经网络基础](https://www.bilibili.com/video/BV1AK4y1P7vs/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_deep-learning-computation\model-construction.ipynb](chapter_deep-learning-computation\model-construction.ipynb)

- [17 使用和购买 GPU](https://www.bilibili.com/video/BV1z5411c7C1/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_deep-learning-computation\use-gpu.ipynb](chapter_deep-learning-computation\use-gpu.ipynb)

- [18 预测房价竟赛总结](): [chapter_multilayer-perceptrons\kaggle-house-price.ipynb](chapter_multilayer-perceptrons\kaggle-house-price.ipynb)

- [19 卷积层](): [chapter_convolutional-neural-networks\conv-layer.ipynb](chapter_convolutional-neural-networks\conv-layer.ipynb)

- [20 卷积层里的填充和步幅](): [chapter_convolutional-neural-networks\padding-and-strides.ipynb](chapter_convolutional-neural-networks\padding-and-strides.ipynb)

- [21 卷积层里的多输入多输出通道](): [chapter_convolutional-neural-networks\channels.ipynb](chapter_convolutional-neural-networks\channels.ipynb)

- [22 池化层](): [chapter_convolutional-neural-networks\pooling.ipynb](chapter_convolutional-neural-networks\pooling.ipynb)

- [23 经典卷积神经网络 LeNet](): [chapter_convolutional-neural-networks\lenet.ipynb](chapter_convolutional-neural-networks\lenet.ipynb) 

- [24 深度卷积神经网络 AlexNet](): [chapter_convolutional-modern\alexnet.ipynb](chapter_convolutional-modern\alexnet.ipynb)

- [25 使用块的网络 VGG](): [chapter_convolutional-modern\vgg.ipynb](chapter_convolutional-modern\vgg.ipynb)

- [26 网络中的网络 NiN](): [chapter_convolutional-modern\nin.ipynb](chapter_convolutional-modern\nin.ipynb)

- [27 含并行连结的网络 GoogLeNe/Inception V3](): [chapter_convolutional-modern\googlenet.ipynb](chapter_convolutional-modern\googlenet.ipynb)

- [28 批量归一化](): [chapter_convolutional-modern\batch-norm.ipynb](chapter_convolutional-modern\batch-norm.ipynb)

- [29 残差网络 ResNet](): [chapter_convolutional-modern\resnet.ipynb](chapter_convolutional-modern\resnet.ipynb)

- [稠密连接网络 DenseNet: chapter_convolutional-modern\densenet.ipynb](chapter_convolutional-modern\densenet.ipynb)

- [30 第二部分完结竞赛:图片分类](https://www.bilibili.com/video/BV1z64y1o7iz/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [叶片分类](https://www.kaggle.com/c/classify-leaves)

- [31 深度学习硬件:CPU 和 GPU](https://www.bilibili.com/video/BV1TU4y1j7Wd/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_computational-performance\hardware.ipynb](chapter_computational-performance\hardware.ipynb)

- [32 深度学习硬件:TPU和其他](https://www.bilibili.com/video/BV1VV41147PC/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_computational-performance\hardware.ipynb](chapter_computational-performance\hardware.ipynb)

- [33 单机多卡并行](https://www.bilibili.com/video/BV1vU4y1V7rd/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_computational-performance\multiple-gpus.ipynb](chapter_computational-performance\multiple-gpus.ipynb)

- [34 多GPU训练实现](https://www.bilibili.com/video/BV1MQ4y1R7Qg/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_computational-performance\multiple-gpus-concise.ipynb](chapter_computational-performance\multiple-gpus-concise.ipynb)

- [35 分布式训练](https://www.bilibili.com/video/BV1jU4y1G7iu/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_computational-performance\auto-parallelism.ipynb](chapter_computational-performance\auto-parallelism.ipynb)

- [36 数据增广](https://www.bilibili.com/video/BV17y4y1g76q/?spm_id_from=333.999.0.0&vd_source=c4c3979529777bf3f5a1d518dcabcdb0): [chapter_computer-vision\image-augmentation.ipynb](chapter_computer-vision\image-augmentation.ipynb)

## 相关资源

- 教材：[《动手学深度学习》第二版](https://zh-v2.d2l.ai/)
- B站视频链接：[【完结】动手学深度学习 PyTorch版](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497)
- 官方代码仓库：[github.com/d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)
- 讨论区：[discuss.d2l.ai/c/chinese-version/](https://discuss.d2l.ai/c/chinese-version/16)

