# 单元 1: 扩散模型简介

欢迎来到Hugging Face扩散模型课程第一单元！在本单元中，您将学习有关扩散模型如何工作的基础知识，以及如何使用🤗 diffusers库。

## 开始本单元 :rocket:

以下是本单元的学习步骤:

- 请确保你已经[注册了该课程](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162)。这样当有新课程材料发布的时候你就会收到通知
- 通读下面的介绍材料以及任何你感兴趣的其他资源
- 查看下面的 _**扩散器简介**_ 笔记本，以使用diffuers库将理论应用到实践中
- 使用notebook或链接的训练脚本来训练和分享您自己的扩散模型
- (可选) 如果您有兴趣看到一个极简的从头开始的项目实现，并探索所涉及的不同设计决策，您可以深入研究 _**从头开始一个扩散模型**_ notebook。


:loudspeaker: 请不要忘了加入我们的频道 [Discord](https://huggingface.co/join/discord), 你可以在`#diffusion-models-class` 频道来讨论课程内容以及分享您的作品。
 
## 什么是扩散模型?

扩散模型是“生成模型”算法家族的新成员通过学习给定的训练样本，生成模型可以学会如何**生成**数据，比如生成图片或者声音。一个好的生成模型能生成一组**样式不同**的输出。这些输出会与训练数据相似，但不是一模一样的副本。扩散模型如何实现这一点？为了便于说明，让我们先看看图像生成的案例。

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> 图片来源于DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>

扩散模型成功的秘诀在于扩散过程的迭代本质。最先生成的只是一组随机噪声，但经过若干步骤逐渐改善，有意义的图像将最终会出现。在每一步中，模型都会估计如何从当前的输入生成完全去噪的结果。因为我们在每一步都只做了一个小小的变动，所以在早期阶段（预测最终输出实际上非常困难），这个估计中的任何error都可以在以后的更新中得到纠正。

与其他类型的生成模型相比，训练扩散模型相对较为容易。我们只需要重复以下步骤即可：
1) Load in some images from the training data
2) Add noise, in different amounts. Remember, we want the model to do a good job estimating how to 'fix' (denoise) both extremely noisy images and images that are close to perfect.
3) Feed the noisy versions of the inputs into the model
4) Evaluate how well the model does at denoising these inputs
5) Use this information to update the model weights

1) 从训练数据中加载一些图像
2) 添加不同级别的噪声。请记住，我们希望模型在面对添加了极端噪声和几乎没有添加噪声的带噪图像时，都能够很好地估计如何“修复”（去噪）。
3) 将带噪输入送入模型中
4) 评估模型对这些输入进行去噪的效果
5) 使用此信息更新模型权重

To generate new images with a trained model, we begin with a completely random input and repeatedly feed it through the model, updating it each time by a small amount based on the model prediction. As we'll see, there are a number of sampling methods that try to streamline this process so that we can generate good images with as few steps as possible.

We will show each of these steps in detail in the hands-on notebooks here in unit 1. In unit 2, we will look at how this process can be modified to add additional control over the model outputs through extra conditioning (such as a class label) or with techniques such as guidance. And units 3 and 4 will explore an extremely powerful diffusion model called Stable Diffusion, which can generate images given text descriptions.  

## Hands-On Notebooks

At this point, you know enough to get started with the accompanying notebooks! The two notebooks here come at the same idea in different ways. 
 
| Chapter                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction to Diffusers                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              |
| Diffusion Models from Scratch                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              |

In _**Introduction to Diffusers**_, we show the different steps described above using building blocks from the diffusers library. You'll quickly see how to create, train and sample your own diffusion models on whatever data you choose. By the end of the notebook, you'll be able to read and modify the example training script to train diffusion models and share them with the world! This notebook also introduces the main exercise associated with this unit, where we will collectively attempt to figure out good 'training recipes' for diffusion models at different scales - see the next section for more info.

In _**Diffusion Models from Scratch**_ we show those same steps (adding noise to data, creating a model, training and sampling) but implemented from scratch in PyTorch as simply as possible. Then we compare this 'toy example' with the diffusers version, noting how the two differ and where improvements have been made. The goal here is to gain familiarity with the different components and the design decisions that go into them, so that when you look at a new implementation you can quickly identify the key ideas.

## Project Time

Now that you've got the basics down, have a go at training one or more diffusion models! Some suggestions are included at the end of the _**Introduction to Diffusers**_ notebook. Make sure to share your results, training recipes and findings with the community so that we can collectively figure out the best ways to train these models.

## Some Additional Resources
 
[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) is a very in-depth walk-through of the code and theory behind DDPMs with 
 maths and code showing all the different components. It also links to a number of papers for further reading.
 
Hugging Face documentation on [Unconditional Image-Generation
](https://huggingface.co/docs/diffusers/training/unconditional_training) for some examples of how to train diffusion models using the official training example script, including code showing how to create your own dataset. 

AI Coffee Break video on Diffusion Models: https://www.youtube.com/watch?v=344w5h24-h8

Yannic Kilcher Video on DDPMs: https://www.youtube.com/watch?v=W-O7AZNzbzQ

你找到了更多很棒的资源？请务必让我们知道，我们会将他们添加到此列表中。
