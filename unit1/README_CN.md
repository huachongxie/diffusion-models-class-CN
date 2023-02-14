# 单元 1: 扩散模型简介

欢迎来到 Hugging Face 扩散模型课程第一单元！在本单元中，你将学习到有关扩散模型如何运作的基础知识，同时也会学到如何使用 🤗 diffusers 库。

## 开始本单元 :rocket:

以下是本单元的学习步骤:

- 请确保你已经 [注册了该课程](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162)。这样当有新课程材料发布的时候你就会收到通知
- 通读下面的介绍材料以及任何你感兴趣的其他资源
- 查看下面的 _**Introduction to Diffusers**_ Notebook 链接，以使用 `diffuers` 库将理论应用到实践中
- 使用 Notebook 或链接的训练脚本来训练和分享你自己的扩散模型
- (可选) 如果你有兴趣看到一个极简的从头开始的项目实现，并探索所涉及的不同设计决策，你可以深入研究 _**Diffusion Models from Scratch**_ 这个 Notebook


:loudspeaker: 请不要忘了加入我们的频道 [Discord](https://huggingface.co/join/discord), 你可以在 `#diffusion-models-class` 频道来讨论课程内容以及分享你的作品。
 
## 什么是扩散模型？

扩散模型是「生成模型」算法家族的新成员通过学习给定的训练样本，生成模型可以学会如何 **生成** 数据，比如生成图片或者声音。一个好的生成模型能生成一组 **样式不同** 的输出。这些输出会与训练数据相似，但不是一模一样的副本。扩散模型如何实现这一点？为了便于说明，让我们先看看图像生成的案例。

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> 图片来源于 DDPM paper (https://arxiv.org/abs/2006.11239)。</em>
<p>

扩散模型成功的秘诀在于扩散过程的迭代本质。最先生成的只是一组随机噪声，但是经过若干步的逐渐改善之后，最终会出现有意义的图像。在每一步中，模型都会估计如何从当前的输入生成完全去噪的结果。因为我们在每一步都只做了一个小小的变动，所以在早期阶段（预测最终输出实际上非常困难），这个估计中的任何 error 都可以在以后的更新中得到纠正。

与其他类型的生成模型相比，训练扩散模型相对较为容易。我们只需要重复以下步骤即可：

1) 从训练数据中加载一些图像
2) 添加不同级别的噪声。请记住，我们希望模型在面对添加了极端噪声和几乎没有添加噪声的带噪图像时，都能够很好地估计如何 “修复”（去噪）。
3) 将带噪输入送入模型中
4) 评估模型对这些输入进行去噪的效果
5) 使用此信息更新模型权重

为了用训练好的模型生成新的图像，我们从完全随机的输入开始，反复将其输入模型，每次根据模型预测进行少量更新。我们之后会学到有许多采样方法试图简化这个过程，以便我们可以用尽可能少的步骤生成好的图像。
我们将在第一单元的实践笔记本中详细介绍这些步骤。在第二单元中，我们将了解如何修改此过程，来通过额外的条件（例如类标签）或使用指导等技术来增加对模型输出的额外控制。第三单元和第四单元将探索一种非常强大的扩散模型，称为稳定扩散 (stable diffusion)，它可以生成给定文本描述的图像。

## 实践笔记本

到这里，你已经足够了解如何开始使用附带的笔记本了！这里的两个笔记本以不同的方式表达了相同的想法。

| Chapter                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction to Diffusers                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darcula1993/diffusion-models-class-CN/blob/main/unit1/01_introduction_to_diffusers_CN.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers_CN.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/darcula1993/diffusion-models-class-CN/blob/main/unit1/01_introduction_to_diffusers_CN.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/darcula1993/diffusion-models-class-CN/blob/main/unit1/01_introduction_to_diffusers_CN.ipynb)              |
| Diffusion Models from Scratch                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darcula1993/diffusion-models-class-CN/blob/main/unit1/02_diffusion_models_from_scratch_CN.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch_CN.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/darcula1993/diffusion-models-class-CN/blob/main/unit1/02_diffusion_models_from_scratch_CN.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/darcula1993/diffusion-models-class-CN/blob/main/unit1/02_diffusion_models_from_scratch_CN.ipynb)              |

在 _**Introduction to Diffusers**_ 这个 Notebook 中，我们使用 diffusers 库中的构造模块显示了与上述不同的步骤。你将很快看到如何根据你选择的任何数据创建、训练和采样你自己的扩散模型。 在笔记本结束时，你将能够阅读和修改示例训练脚本，以训练扩散模型，并将其与全世界共同分享！ 本笔记本还介绍了与本单元相关的主要练习，在这里，我们将共同尝试为不同规模的扩散模型找出好的「训练脚本」- 请参阅下一节了解更多信息。

在 _**Diffusion Models from Scratch**_ 这个 Notebook 中，我们展示了相同的步骤（向数据添加噪声、创建模型、训练和采样），并尽可能简单地在 PyTorch 中从头开始实现。然后，我们将这个「玩具示例」与 `diffusers` 版本进行比较，并关注两者的区别以及改进之处。这里的目标是熟悉不同的组件和其中的设计决策，以便在查看新的实现时能够快速确定关键思想。

## 项目时间

现在，你已经掌握了基本知识，可以开始训练你自己的扩散模型了！ _**Introduction to Diffusers**_ 这个 Notebook 的末尾有一些小提示，希望你能与社区分享你的成果、训练脚本和发现，以便我们能够一起找出训练这些模型的最佳方法。

## 一些额外的材料

- [《Hugging Face 博客: 带注释的扩散模型》](https://huggingface.co/blog/annotated-diffusion)是对 DDPM 背后的代码和理论的非常深入的介绍，其中包括数学和显示了所有不同的组件的代码。它还链接了一些论文供进一步阅读:
- [Hugging Face 文档: 无条件图像生成 (Unconditional Image-Generation)](https://huggingface.co/docs/diffusers/training/unconditional_training)，包含了有关如何使用官方训练示例脚本训练扩散模型的一些示例，包括演示如何创建自己的数据集的代码：
- AI Coffee Break video on Diffusion Models: https://www.youtube.com/watch?v=344w5h24-h8
- Yannic Kilcher Video on DDPMs: https://www.youtube.com/watch?v=W-O7AZNzbzQ

发现了有其他任何有帮助的资源？请 [向我们提出](https://github.com/huggingface/diffusion-models-class/issues)，我们会将其加入到上面的列表中。