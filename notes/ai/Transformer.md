# [25.11.8]LLM模型训练计算过程

## 零、词表准备
目前，有很多种方法生成词表，不同方法由于对词的切分、排序逻辑有差异，在词表长度、粒度、内容等方面均有差异。

词表可以理解成一个字典，内容是每一个Token的内容和索引序号的映射关系，比如 a:1，b:2...

## 一、输入处理
### 1.1 输入内容的tokenization
输入的内容由词语组成，所以第一步需要分词，将内容按词表拆分为token序列，拆分策略可能是按词、按字符、按子词等。
比如按照字词分词策略，“hello world”可能拆分为he-l-lo wor-l-d的token序列。

如果词表中：
- he->101
- llo->102
- wor->104
- ld->105
- 空格->100
- 特殊token，eos->200

对于拆分之后的token序列，按照词表可以映射成token的序号，并按需添加一些特殊token，即[101,102,100,104,105,200]这样的token_id序列。
这些token_id序列可以根据词表很方便的decode成原始的词序列。

### 1.2 Token嵌入
输入内容薛烈在tokenization之后已经变成了token_id序列，Token嵌入会将Token_id序列中每个元素变成一个d维（隐藏维度）的向量，即对于[101,102,100,104,105,200]这样长度为6的序列，会转换成6*d的矩阵。

从token_id到d维向量的计算过程会分别计算词嵌入向量和位置编码，两者加起来形成最终的嵌入向量：
1. token_id经过Embedding Layer嵌入层计算，转换为词嵌入向量，
假设d为4，101->[1,2,3,4] (值是编的)
Embedding计算实际上是把词表中所有的Token_id映射到d维向量空间中，在向量空间中表示各个Token之间的关系，**整个Embedding权重矩阵（嵌入矩阵）本质上就是一个可训练的token_id到嵌入向量的字典（查找表）**。
- 嵌入矩阵中的值会在训练过程中改变、调整，以使得语义有关联的token的词嵌入向量在嵌入空间的位置关系可以表达关联；
  - 初始状态下，嵌入矩阵中的值可以是随机的或者预设的；
  - 在训练中，损失函数对模型参数矩阵W、嵌入矩阵E分别求梯度，通过梯度下降同时更新嵌入矩阵和参数矩阵；
  - 嵌入矩阵在训练中确定，在推理中只需要对每个Token_id查询即可，不需要重复计算；

对于输入的n个Token的token_id，假设词表长度为m，可以生成一个输入的`n*m`矩阵，矩阵中行向量表示各个Token，向量中Token在词表中对应位置数值为1，其余为位置数值为0。
- 输入的矩阵无法在训练的反向传播中变化，调整的是权重矩阵。
- 权重矩阵中只有token对应行的数值会被调整，其余部分无法计算梯度，不会被更新。

假设维度为d，Embedding权重矩阵则为`m*d`的矩阵，即词表中每个Token对矩阵中的一行。

在输入矩阵与权重矩阵相乘后，输出`n*d`的Embedding结果，通过输入矩阵中每行中数值为1的元素选中权重矩阵中的对应行，即输出结果中的每一行对应token的d维词嵌入向量。

2. token_id在整个序列中的位置经过计算生成向量来标识位置，通常使用正弦、余弦来编码的，比如旋转位置编码，位置0->{0.1,0.1,0.1,0.1} (值也是编的)

# 二、训练
## 2.1 前置概念
在当前的训练中，有一些公认的概念：
- Epoch，轮次，也就是说整个训练数据集中的每个训练样本都通过了模型一次，即数据集有10K个样本，在Epoch中10K个样本都在模型中训练了1次；
- Batch或者mini-batch，批，一个Epoch中的数量往往会很大，无法一次性将所有样本输入模型，只能将样本分批输入，每批包含batch_size个样本，这样的一批就是Batch；
  - 比如Epoch中有10K样本，每个batch可以输入50个样本，这样这个Epoch会分为200个Batch；
  - 通常，一个训练数据集的不同epoch对batch的划分不固定，epoch会对数据进行shuffle，或者按照策略重新划分，即不同的epoch中同序号的batch包含的样本完全不一样，或者不完全一样；
    - 训练过程中数据的shuffle可以有效避免模型学习到样本出现的固定模式，防止模型过拟合；
- Iteration，迭代，每完成一个batch的训练成为一次iteration

参数更新（Embedding权重举证和模型参数矩阵）的时机可以有不同的策略：
- 每个样本处理完更新参数，更新的频率会很高，由于下一个样本的处理以来于本次样本的参数更新，样本的处理只能串行，整体训练速度会慢，且更新的方向完全受单个样本影响，不稳定；
- epoch结束时，对所有样本的更新求均值更新参数，这样在更新之前需要存储所有样本的更新，内存需求大，但是更新的方向会比较稳定；
- batch结束时，对batch中的所有样本产生的参数更新取均值更新参数，这样平衡了按照样本粒度更新和按照epoch粒度更新，当前通常是采用这种策略；

## 2.2 训练过程
对于1个batch的输入，在转换为嵌入词向量并叠加位置编码之后，最终生成形状为`batch_size*input_token_length*embedding_dim`的张量。

TODO Transformer的结构和处理整体过程


### 2.2.1 Transformer

#### 2.2.1.1 Transformer架构
Transformer中由Encoder和Decoder的Block组成，输入的Token矩阵会依次被每个Encoder处理，之后再被Decoder逐层处理。
这里说的只是Attention论文中的设计，是Transformer的基础框架。各种大模型在结构层面可能调整，可能只有Encoder或者只有Decoder，Decoder-Only的模型缺少Encoder的输出，交叉Attention无法计算，Decoder中交叉注意力计算步骤可能省去或者改变。此外，Encoder和Decoder之间的连接关系也可能发生变化。


#### 2.2.1.2 Encoder Block
Transformer由多层结构叠加，每层成为一个Block，以下是1个Block
1. 多头自注意力

注意力QKV的一种存疑的解释是：

每个注意力头本质上捕捉某种特征关系；
$W_Q$描述特征关系
$W_K$描述token的性质，Q、K相乘可获得每个Token与其他Token的方向一致性，或匹配程度，相关性较高的Token之间的点积相对较大；
点积数值的相对关系相比数值本身更加重要，所以使用softmax操作将值变为0~1之间概率，一个Token相对其他Token的所有数值之和为1；


Q\K\V 计算过程
对于张量`batch_size*input_token_length*embedding_dim`每个Xi为`input_token_length*embedding_dim`的矩阵，

`Q=Xi*Wq`，Wq矩阵为 `embedding_dim*hidden_dim`，通常`hidden_dim`相比`embedding_dim`小很多，每个token计算之后对应一个维度为`hidden_dim`的向量，多个注意力头处理后的向量拼接后形成最终向量，维度上为embedding_dim；

`hidden_dim = embedding_dim\num_heads`

多个注意力头的权重矩阵实际上在一个大矩阵中存储，形状为`embedding_dim*embdding_dim`，每个注意力头都是一个切片

`K=Xi*Wk`

`V=Xi*Wv`，$W_V$比较大（`embedding_dim*embedding_dim`），通常分解成两个矩阵（$V_{up}$和$V_{down}$）相乘来优化存储空间，$V_{up}$被当做$W_V$（`hidden_dim*embedding_dim`），而所有$V_{down}$整合在一起称作输出矩阵

每个注意力头均会处理整个输入矩阵，处理后输出的向量维度为`hidden_dim=embedding_dim / num_heads`。

每个注意力头分别计算注意力$Attention(Q,K,V)=softmax(\frac{Q*K^T}{\sqrt{K_dim}})*V$
多个注意力头的注意力输入拼接，再通过线性变化得到多头自注意力的最终输出，即$Output=[\begin{matrix} Attention_1 & Attention_2... & Attention_n \end{matrix}]*[\begin{matrix}W^o\end{matrix}]$。

2. 残差连接与归一化：多头注意力输出叠加输入X，进行层归一化；
这里的归一化是后归一化，即在处理之后进行归一化，也可以选择预归一化，即先归一化再进入子层，GPT通常选择预归一化策略；

- 预归一化处理策略（Pre-LayerNorm）：输入X层归一化-->多头注意力计算-->残差连接`X=X+Attention(LN(X))`-->输入X层归一化-->前馈神经网络-->残差连接`X=X+FFN(LN(X))`
- 后归一化处理策略（Post-LayerNorm）：多头注意力计算-->残差连接和归一化`X=LN（X+Attention(X)）`-->前馈神经网络-->残差连接和归一化`X=LN（X+FFN(X)）`


TODO adnrej karpathy课程


TODO 矩阵Wq、Wk、Wv是静态的么，不会随训练发生变化？值是随机的么？这三个矩阵的维度是多少，是embedding_dim*X？

输入张量中的每个矩阵`[input_token_length*embedding_dim]`

3. 前向神经网络FFN？
通过两个线性变换，中间有一个激活函数ReLU或者GELU，$X = Linear_2(ReLU(Linear_1(X)))+X$，这里两个线性层的参数量占到了整个LLM参数量的$\frac{{2}}{3}$

第一个线性变换hidden_dim扩大到intermediate_size，通常是hedden_dim的4倍，
$Linear_1(X)=[W_{up}]*X + B_{up}$,B为偏置向量；
ReLU是下线性整流函数的缩写，其实就是负值归0，正值不变；

第二个线性变换将维度缩小回hidden_dim，$Linear_2(X)=[W_{down}]*X + B_{down}$,


1. 再次残差连接与归一化

**上述步骤为Transformer中1个Encoder Block的处理逻辑**

#### 2.2.1.3 Decoder Block
TODO 解码器内部的架构和流程

交叉注意力的计算过程与自注意力计算逻辑一致，不同的是，KV由编码器计算输出，Q矩阵由解码器计算输出；

### 2.2.2 输出层
TODO 线性变换 + Softmax概率 -> 按照词表选择最终输出的token

- encoder、decoder中block是什么
TODO Linear如何计算，意义是什么
TODO Softmax如何计算，意义是什么
TODO Norm如何计算，意义是什么





## 2.3 梯度下降、反向传播

## 2.4 分布式训练


## 数学公式语法
行间公式
$$\sum_{i=1}=\frac{{A+b}}{c} * \sqrt{X_i}$$
$$
\begin{matrix}
a & b \\
c & d
\end{matrix}
$$

$$
f(x) = \begin{cases} 
x + 1, & \text{if $x < 0$} \\
x^2, & \text{if $x \geq 0$}
\end{cases}
$$




行内公式: $X_i+Y^i$
$\begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix}$