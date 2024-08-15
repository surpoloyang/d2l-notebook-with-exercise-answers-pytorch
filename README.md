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

由于d2l各个版本的torch实现都缺少部分函数与类的定义，本仓库以`1.0.3`版本的d2l库torch实现为基础进行修改与完善，敬请期待！

## 目录

待完善。。。

## 相关资源

- 教材：[《动手学深度学习》第二版](https://zh-v2.d2l.ai/)
- B站视频链接：[【完结】动手学深度学习 PyTorch版](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497)
- 官方代码仓库：[github.com/d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)
- 讨论区：[discuss.d2l.ai/c/chinese-version/](https://discuss.d2l.ai/c/chinese-version/16)

