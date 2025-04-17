# PyTorch 官方教程中文版 (Jupyter Notebook)

## 简介

本项目包含一系列 Jupyter Notebook 文件，是 PyTorch 官方入门教程的中文翻译和实践。这些教程旨在帮助初学者理解 PyTorch 的核心概念，并掌握使用 PyTorch 构建和训练神经网络的基本流程。

内容涵盖：

*   **PyTorch 基础**: Tensors 操作 ([`1-Brief Introduction/1 - PyTorch Tensors.ipynb`](1-Brief%20Introduction/1%20-%20PyTorch%20Tensors.ipynb), [`2-Tensors.ipynb`](2-Tensors.ipynb))
*   **自动微分 (Autograd)**: 理解梯度计算机制 ([`3-Autograd.ipynb`](3-Autograd.ipynb))
*   **构建模型**: 使用 `torch.nn.Module` 定义网络结构 ([`1-Brief Introduction/2 - A Simple PyTorch model.ipynb`](1-Brief%20Introduction/2%20-%20A%20Simple%20PyTorch%20model.ipynb), [`4-Building Models.ipynb`](4-Building%20Models.ipynb))
*   **数据处理**: `Dataset` 和 `DataLoader` 的使用 ([`1-Brief Introduction/3 - Dataset and DataLoader.ipynb`](1-Brief%20Introduction/3%20-%20Dataset%20and%20DataLoader.ipynb))
*   **模型训练**: 实现完整的训练循环 ([`1-Brief Introduction/4 - A Simple PyTorch Training Loop.ipynb`](1-Brief%20Introduction/4%20-%20A%20Simple%20PyTorch%20Training%20Loop.ipynb), [`6-Model Training with PyTorch.ipynb`](6-Model%20Training%20with%20PyTorch.ipynb))
*   **可视化**: 使用 TensorBoard 监控训练过程 ([`5-Tensorboard Support.ipynb`](5-Tensorboard%20Support.ipynb))
*   **模型可解释性**: Captum 入门 ([`7-Captum/Getting-Started-with-Captum.ipynb`](7-Captum/Getting-Started-with-Captum.ipynb))

相应的英文视频教程可以在 YouTube 上观看：[PyTorch Basics - YouTube](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=29s)

## 内容列表

*   **`1-Brief Introduction/`**: 快速入门系列
    *   [`1 - PyTorch Tensors.ipynb`](1-Brief%20Introduction/1%20-%20PyTorch%20Tensors.ipynb): PyTorch 张量基础
    *   [`2 - A Simple PyTorch model.ipynb`](1-Brief%20Introduction/2%20-%20A%20Simple%20PyTorch%20model.ipynb): 构建一个简单的 PyTorch 模型
    *   [`3 - Dataset and DataLoader.ipynb`](1-Brief%20Introduction/3%20-%20Dataset%20and%20DataLoader.ipynb): 数据集和数据加载器
    *   [`4 - A Simple PyTorch Training Loop.ipynb`](1-Brief%20Introduction/4%20-%20A%20Simple%20PyTorch%20Training%20Loop.ipynb): 一个简单的 PyTorch 训练循环
*   [`2-Tensors.ipynb`](2-Tensors.ipynb): PyTorch 张量详解
*   [`3-Autograd.ipynb`](3-Autograd.ipynb): 自动微分机制
*   [`4-Building Models.ipynb`](4-Building%20Models.ipynb): 在 PyTorch 中构建模型
*   [`5-Tensorboard Support.ipynb`](5-Tensorboard%20Support.ipynb): TensorBoard 可视化支持
*   [`6-Model Training with PyTorch.ipynb`](6-Model%20Training%20with%20PyTorch.ipynb): 使用 PyTorch 训练模型（FashionMNIST 示例）
*   **`7-Captum/`**: 模型可解释性
    *   [`Getting-Started-with-Captum.ipynb`](7-Captum/Getting-Started-with-Captum.ipynb): Captum 入门

## 用法

### 环境要求

确保您已安装以下库：

*   `torch`
*   `torchvision`
*   `matplotlib`
*   `numpy`
*   `tensorboard`

可以使用 `pip` 或 `conda` 进行安装：

```bash
# 使用 pip
pip install torch torchvision matplotlib numpy tensorboard

# 或者使用 conda
conda install pytorch torchvision matplotlib numpy tensorboard -c pytorch