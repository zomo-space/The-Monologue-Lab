# AI学习计划

之前对于大模型相关技术有所了解，甚至在大模型火爆之前就对AI相关技术有所涉猎。但是，大模型爆发式发展之后，本就不成体系的知识变得更加零散了，甚至有些是相互冲突的。

所以，我又想了一下我在AI方面自己要学习并掌握内容，并且采用费曼学习法，输出自己的理解，原则是：
<p style="text-indent:2em">只记录自己的理解，不做原文摘要、笔记</p>

## 大模型相关基础
这里的学习思路是从按照B站、Youtube上面的视频课程学习，目标快速构建基础，并识别后面几个领域的关键学习点。
主要课程和资料如下：

- 李宏毅机器学习2025​：包含大模型最新内容（中文）
- 斯坦福CS324 - Large Language Models：系统介绍大模型的基础知识与前沿研究。
- Coursera吴恩达机器学习课程
- 哈佛CS324​：大模型基础理论（英文）
- youtube视频 let's build gpt: from scratch, in code, spelled out
- 关键论文精读：Transformer论文精读班（如李沐等人的系列解读）


## 大模型训练相关
关于大模型训练，需要学习训练的整个过程，并从小模型（minimind）入手在本地完成训练过程（tinyllama也可以考虑？）。
有几个问题一直没有彻底弄清：
- 计算图到底是什么粒度的，任务还是算子？
- 专家并行的专家为什么会出现，和其他并行有什么关系？

学习分布式训练中各种并行技术：数据并行、模型并行、流水线并行，除了要学习并行的过程，还需要知道为什么会出现某种并行技术，解决的是什么问题。

在此基础上，弄清楚专家并行相关技术。

主要资料如下：
- Understanding the Transformer architecture for neural networks ：https://www.jeremyjordan.me/transformer-architecture/
- 微软出的人工智能系统学习资源，涵盖了全链条，从基础支持、神经网络基本原理和数学知识、AI系统和实践等 https://github.com/microsoft/ai-edu/blob/master/基础教程/A6-人工智能系统/README.md
- 微软出的人工智能系统相关，是上面的第6章，涵盖了深度学习框架、分布式训练和推理相关的很多内容：
https://github.com/microsoft/AI-System/blob/main/Textbook/README.md


## 大模型推理相关

推理方面需要弄明白整个推理的过程，在此基础上理解PD分离相关技术：
- P的过程和D的过程为什么要分开？
- P、D具体的计算过程？
- KV cache引入的意义是什么？

还需要深入分析一下推理框架相关的原理和机制，比如vLLM的源码。


## 底层算子相关
思路弄明白为什么会有这些算子，在什么过程中用到的，看能不能找一些底层算力看一下设计、实现。


## 其他资料
其他一些资源，可以作为补充了解AI发展到现在的历程：
- 详解前馈、卷积和循环神经网络技术 https://zhuanlan.zhihu.com/p/29141828
- 很全的深度学习资源，发展过程中的资源都有，https://github.com/Mikoto10032/DeepLearning?tab=readme-ov-file


