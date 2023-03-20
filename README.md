# <center>这是lbl的gkc试炼

## <center> 第一次试炼
使用SGD优化器的最好参数：

MLP和MLQP:取--hidden_num 1024 --lr 0.002 --momentum 0.9进行比较
MLP最好的情况是在--hiddennum 2048，--lr 0.005
SOTA的最好情况是--hiddennum 64 --lr 0.01 --momentum 0.9


用Adam(效果更好)：
MLQP:
--model MLQP --hidden_num 256 --lr 0.001 
MLP:
--model MLP --hidden_num 2048 --lr 0.001
SOTA:hidden num 仍然用64足矣

运行结果示例：

![](./demo/SOTA_DEMO.png)
# <center> 工程实践与科技创新J

## <center> 第一次作业

### <center> 陈垍铮 520021911182

# Q1：
符号规定：
设神经网络共有M层， $d_{Mk}$表示第M层（输出层）的第k个神经元对应的标签，其中$k\in {\{1,2,...,N_M\}}$,再令第j层神经元的个数为$N_j$, 第j层第k个神经元的输出为$x_{jk}$,激活函数为f

## （1）随机方式下误差反向传播算法的推导
对于最后一层的神经元，有误差$e_{Mk} =d_{Mk}- x_{Mk}$,令经过激活函数之前的输出为：$net_{Mk} = \sum_{i=1}^{N_{k-1}}(u_{M,M-1,i}x_{M-1,i}^2+v_{M,M-1,i}x_{M-1,i}+b_{Mk})$,则有最终的输出$x_{Mk} = f(net_{Mk})$

又知总误差能量为：$\epsilon = \frac{1}{2}\sum_{k\in C}e_k^2$

由链式求导法则：

$$
\frac{\partial \epsilon}{\partial w_{M,M-1,i}} = \frac{\partial \epsilon}{\partial e_{Mk}}\frac{\partial e_{Mk}}{\partial x_{Mk}}\frac{\partial x_{Mk}}{\partial net_{Mk}}\frac{\partial net_{Mk}}{\partial w_{M,M-1,i}}
$$

其中w可以取u，v两个参数，定义：

$$
\delta_{Mk} = e_{Mk}f_{Mk}'(net_{Mk})
$$
则有：
$$
\frac{\partial \epsilon}{\partial u_{M,M-1,i}} = \delta_{Mk}x_{M-1,k}^2
$$

$$
\frac{\partial \epsilon}{\partial v_{M,M-1,i}} = \delta_{Mk}x_{M-1,k}
$$

对于第k层隐层的第j个神经元，其经过激活函数之前的输出为：$net_{kj} = \sum_{i=1}^{N_{k-1}}(u_{kji}x_{k-1,i}^2+v_{kji}x_{k-1,i}+b_{kj})$,此时的链式求导法则可以写作:

$$
\frac{\partial \epsilon}{\partial w_{k-1,j,i}} = \frac{\partial \epsilon}{\partial x_{k,j}}f'(net_{kj})\frac{\partial net_{kj}}{\partial w_{k-1,j,i}}
$$

此时令
$$
\delta_{kj} = \frac{\partial \epsilon}{\partial x_{k,j}}f'(net_{kj})
$$

由$\epsilon = \frac{1}{2}\sum_{i \in C}e_i^2$可知，

上式中
$$\frac{\partial \epsilon}{\partial x_{k,j}}=\sum_{i}e_i\frac{\partial e_i}{\partial x_{k,j}} = \sum_{i}e_i\frac{\partial e_i}{\partial net_{j,i}}\frac{\partial net_{j,i}}{\partial x_{k,j}}
$$ 

可知，$\frac{\partial \epsilon}{\partial x_{k,j}}$可以用$\delta _{k+1}$ 递推表示，表示为:

$$
\frac{\partial \epsilon}{\partial x_{k,j}} = \sum_i \delta_{k+1,i}(2u_{k+1,i,j}x_{kj}+v_{k+1,ij})
$$

从而最终得到：

$$
\delta_{kj} = f'(net_{kj})\sum_{i=1}^{N_{k+1}} \delta_{k+1,i}(2u_{k+1,i,j}x_{kj}+v_{k+1,ij})
$$


故而有：

$$
\frac{\partial \epsilon}{\partial u _{k,j,i}} = \delta_{kj}x_{kj}^2 
$$

以及


$$
\frac{\partial \epsilon}{\partial v _{k,j,i}} = \delta_{kj}x_{kj} 

$$

综上所述，可以采用以下公式对多层二次感知机的权重参数进行更新：

$$
\Delta u_{kji} = \eta_{1}\delta_{kj}x_{k-1,i}^2
$$

$$
\Delta v_{kji} = \eta_{2}\delta_{kj}x_{k-1,i}
$$
其中$\delta_{kj}$分别由以上输出层和隐层的推导公式给出

## （2）批量方式下误差反向传播算法的推导
与随机方式的总误差能量：$\epsilon = \frac{1}{2}\sum_{k\in C}e_k^2$不同，批量方式一次更新采用多个样本误差能量的平均值，即：

$$
\epsilon_{av} = \frac{1}{2N}\sum_{n = 1}^N \sum_{j \in C}e^2_j
$$
对于输出神经元，结果只会改变链式法则的第一项，此时
$$
\frac{\partial \epsilon}{\partial e_{Mk}} = \frac{1}{N}\sum_{n=1}^N e_j(n)
$$
故而局部梯度可以写成：
$$
\delta_{Mk} = \frac{1}{N}\sum_{n=1}^Ne_{Mk}(n)f_{Mk}'(net_{Mk})
$$
对于隐层神经元，
$$
\frac{\partial \epsilon}{\partial x_{k,j}}=\frac{1}{N}\sum_{n=1}^N\sum_{i}e_i(n)\frac{\partial e_i(n)}{\partial x_{k,j}} 
$$

故而最终的更新公式为：

$$
\frac{\partial \epsilon}{\partial u _{k,j,i}} = \frac{1}{N}\sum_{n=1}^N\sum_{i}\delta_{kj}x_{kj}^2

$$

以及

$$
\frac{\partial \epsilon}{\partial v _{k,j,i}} = \frac{1}{N}\sum_{n=1}^N\sum_{i}\delta_{kj}x_{kj}
$$

# Q2 && Q3
## 实验目的
本实验旨在比较MLP和MLQP两种模型在双螺旋数据集上的分类效果，并分析其性能差异。

## 实验数据
实验数据为双螺旋数据集，数据集由训练集和测试集组成，均是从双螺旋形分布的数据样本中采样得到。具体数据格式为每个样本包含两个特征x、y和一个标签label，label可以取1或0。数据集共包含600个样本，其中300个作为训练集，300个作为测试集。

## 实验方法
本实验采用了MLP和MLQP两种模型进行分类任务。实验代码已经上传到
https://github.com/Otsuts/lbl1

其中为了对比两种模型的优劣和模型结构对分类结果的影响，共实现了三种模型，代码使用torch实现这三种模型，并运用matplotlib绘制出最后模型的决策面，另外在优化器的选择上，通过尝试证明了Adam优化器效果比SGD更好，因此以下实验中优化器选择Adam。

* MLP：采用单隐层的多层感知机（Multi-Layer Perceptron），其中隐层的神经元个数为128，隐层激活函数为ReLU，输出层激活函数为Sigmoid。损失函数采用二元交叉熵损失函数，优化器为Adam，学习率取0.001，0.05，0.1，取训练轮次为1000轮。
* MLQP：采用含有一层隐藏单元的多层二次感知机（Multi-Layer Quadratic Perceptron），其中隐层的神经元个数为128，隐层激活函数为ReLU，输出层激活函数为Sigmoid。损失函数采用二元交叉熵损失函数，优化器为Adam，学习率取0.001，0.05，0.1，训练轮次为1000轮。
* SOTA：两层隐藏单元的多层感知机（Multi-Layer Perceptron），其中隐层的神经元个数均为64，隐层激活函数为ReLU，输出层激活函数为Sigmoid。损失函数采用二元交叉熵损失函数，优化器为Adam，训练轮次为1000轮。采用SOTA模型是为了首先取得一个比较好的效果，通过调参使得MLP和MLQP的效果逼近该模型，从而方便确定最合适的参数。SOTA模型也说明了，加深神经网络比加宽神经网络更有效，更能提高模型的表达能力。
## 实验结果
### SOTA模型结果
SOTA模型的参数取--hiddennum 64 --lr 0.01，已经可以得到100%的测试集准确率，其决策面的绘制效果如下图所示：

![](./pictures/SOTA.png)

### MLP模型结果
在0.001，0.01，0.1的学习率下，训练集和测试集最高的准确率以及训练所用的时间如下图所示

| 模型学习率     | 训练集准确率 | 测试集准确率 |训练时间（cpu）/s
| ----------- | ----------- | ----------- |-----------|
| 0.001     | 69.0000      |     71.0000     |  8.33      |
| 0.005   | 77.6667   |      77.6667    |    8.79         |
| 0.01   | 81.3333    |      82.6667    |        8.33     |

![](./pictures/MLP.png)

### MLQP模型结果
MLQP模型在训练集和测试集上的分类结果如下表所示：
| 模型学习率     | 训练集准确率 | 测试集准确率 |训练时间|
| ----------- | ----------- | ----------- |---------|
| 0.001     | 96.6667      |     98.0000     | 12.89     |
| 0.005   | 100.0000   |      100.0000    |       12.66   |
| 0.01   | 100.0000   |      100.0000   |        12.72    |



![](./pictures/MLQP.png)


## 结论分析
从实验结果中，我们可以得出以下结论：
1. 从学习率对训练结果的角度来看，各模型在学习率适当增大（从0.001到0.01）的过程中的训练效果都越来越好。说明对于双螺旋分类这样一类简单的任务，更适合于使用较大的学习率来使模型收敛更快。
2. 从模型结构的角度来看，使用具有非线性映射能力的MLQP能够极大的提升模型的表达能力，同时取得了不错的时间性能，在结果相同（均为100%）的前提下与两层隐层的多层感知机相比时间开销还要少2s
3. 比起增加神经网络的宽度，显然增多层数对于表达能力的提高更加有效，因此MLQP虽然具有一定的创新性，但其最终性能和效果并非不可代替。

## <center> 第二次试炼
使用SVM进行SEED-IV情感数据集分类

运行方法：

```
python main.py --method external\internal --category none\ovr --C 200 --kernel rbf -- degree
```

其中method指定了是跨被试还是被试独立，category指定使用自己写的多分类器还是自带的，C,kernel和degree是SVM的参数


# <center>实验报告：基于SVM的SEEDIV多类情绪分类
## <center> 陈垍铮 520021911182

### 1.概述：

在这个实验中，我们使用支持向量机（SVM）实现了SEEDIV数据集的多类情绪分类。SEEDIV数据集包含15名被试，每名被试都做了3次实验，每次实验有24段视频，一共四类情绪。我们取前16段视频为训练集，后8段视频为测试集。在被试依赖条件下，我们训练了每个session的每名被试单独的模型，即一共训练了45个模型，最终取平均准确率。在跨被试条件下，我们将所有被试3个session的数据拼接，进行留一交叉验证，最终取平均准确率。

### 2.实验步骤：

#### 1.数据预处理

对于原始的数据，我们采用两种方式进行实验结果的对比：

1. 不进行预处理，直接有SVM进行多类情绪分类
2. 使用sklearn 中的StandardScaler工具对数据进行归一化

对于原始的数据，从npy中读得的文件是一个n\*62\*5的向量，我们每个数据62\*5的矩阵进行拼接，使得每个数据由一个310维的特征向量表征，并将此特征向量送入SVM进行分类。

#### 2.被试依赖条件下的多类情绪分类

在被试依赖条件下，我训练了每个session的每名被试单独的模型。我使用Python中的Scikit-learn库实现了SVM分类器，并且使用网格搜索法选择最优的超参数。我们使用了径向基函数（RBF）作为核函数，并且设置了不同的C值。最终，我选择了C=200作为最优超参数。最终，我们将45个模型的平均准确率作为被试依赖条件下的多类情绪分类的结果。

我将自己实现的ovr分类方法和SVC库内置的分类方法实验效果进行对比，以验证实现方法的正确性。

#### 3.跨被试条件下的多类情绪分类

在跨被试条件下，我们将所有被试3个session的数据拼接，进行留一交叉验证。具体地，我们将所有被试的所有session的数据拼接在一起，并且随机划分为15个部分，每个部分包含了每个被试的所有session的数据。在每次留一交叉验证中，我们将14个部分的数据作为训练集，1个部分的数据作为测试集。我们使用相同的SVM分类器和超参数将所有测试集的平均准确率作为跨被试条件下的多类情绪分类的结果。

在跨被试条件中，可以发现是否对数据进行归一化预处理对实验结果有较大影响。这是因为每位被试的数据分布有较大差异，均值和方差分布不一致，所以留一交叉验证的效果不好。

实验代码除了已经随附，并也上传至 https://github.com/Otsuts/lbl

### 3.实验结果：

#### 1.被试依赖条件下的实验结果
| 模型     | 是否归一化 | 测试集准确率 |训练时间（cpu）/s
| ----------- | ----------- | ----------- |-----------|
| 内置SVC库    | 否      |     58.6     |  0.57      |
| OVR手动实现  | 否  |     59.7     |   2.31       |
| 内置SVC库    | 是      |     43.1     |  0.62     |
| OVR手动实现  | 是  |     45.0     |   2.65       |


#### 2.跨被试条件下的实验结果
| 模型     | 是否归一化 | 测试集准确率 |训练时间（cpu）/s
| ----------- | ----------- | ----------- |-----------|
| 内置SVC库    | 否      |     41.7     |  308      |
| 内置SVC库    | 是      |     62.4     |  197    |
| OVR手动实现  | 是  |     62.2     |   2538       |


#### 3.分析：
首先我必须感叹一下SVC库代码的运行效率之高，时间性能上远远领先于手动实现的OVR。其次，是否加入归一化对模型效果也有不同程度的影响，对于被试依赖的条件，归一化反而会降低模型性能，而对于跨被试的条件，归一化能够极大提升模型的性能。

值得注意的是，是否加入归一化也会影响模型的收敛速度，对于数据分布有较大差异的模型，不加归一化会使模型收敛速度变慢。

### 4.结论：

在本次实验中，我们使用SVM实现了SEEDIV数据集的多类情绪分类任务。在跨条件下，我们的模型可以取得相对较好的性能，平均准确率达到了62%。然而，在被试依赖条件下，我们的模型表现一般，平均准确率只有59%。并且，跨被试条件下，归一化能够提升模型的表达能力，但是被试依赖条件下模型性能反而会下降。这也是由于多个用户的数据分布差异导致的。而这提示我们在进行SVM数据分类任务时，应该考虑数据的来源和特点，并且在模型训练中采用更加鲁棒的方法，以提高模型的泛化性能。


## <center> 第三次试炼
usage: main.py [-h] [--method {internal,external}]
               [--dataset_path DATASET_PATH] [--scale SCALE]
               [--across_scale ACROSS_SCALE] [--learning_rate LEARNING_RATE]
               [--batch_size BATCH_SIZE] [--epoch_times EPOCH_TIMES]
               [--test_iter TEST_ITER]
               [--convolution_method CONVOLUTION_METHOD]

optional arguments:
  -h, --help            show this help message and exit
  --method {internal,external}
  --dataset_path DATASET_PATH
  --scale SCALE
  --across_scale ACROSS_SCALE
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --epoch_times EPOCH_TIMES
  --test_iter TEST_ITER
  --convolution_method CONVOLUTION_METHOD
  
  
  # <center> 实验报告：基于CNN的SEEDIV多类情绪分类

## <center> 陈垍铮 520021911182

### 1.概述
在本次实验中，我们使用卷积神经网络（CNN）实现了SEEDIV数据集的多类情绪分类。SEEDIV数据集包含15名被试，每名被试都做了3次实验，每次实验有24段视频，一共四类情绪。我们取前16段视频为训练集，后8段视频为测试集。在被试依赖条件下，我们训练了每session的每名被试单独的模型，即一共训练了45个模型，最终取平均准确率。在跨被试条件下，我们将所有被试3个session的数据拼接，进行留一交叉验证，最终取平均准确率。

### 2.实验步骤

#### 1.数据预处理
对于原始的数据，我们对数据预处理的方法与SVM进行情感分类的方法一样，采用两种方式进行实验结果的对比：
1. 不进行数据预处理，直接用SVM进行多类情绪分类
2. 使用sklearn中的StandardScaler工具对数据进行归一化

#### 2.卷积神经网络搭建
在本次实验中，如何设计神经网络的架构成为了取得较好实验结果的关键。本次实验主要存在三个比较大的挑战：
1. **由于在被试独立的实验中，每个被试的数据量较小，只有600多个，模型很容易过拟合，导致训练集上正确率达到100%，但是测试集的准确率很低，因此设计合适的卷积神经网络架构显得非常重要**
2. **跟使用SVM的实验一样，数据分布存在差异的问题仍然存在，因此对数据进行一定的预处理就显得很重要**
3. **神经网络的架构和相关超参数调试困难，每次训练所花的时间较长，并且数据预处理的方式非常灵活，导致整体代码调试难度较大**
   

因此本实验中我用两种不同方式搭建了卷积神经网络：
1. 将每位用户的输入首先拼接成1*310维的向量，再用初始通道数为1的卷积神经网络进行处理。
2. 保留源数据的形式，用初始通道数为5的卷积神经网络进行处理。

两个神经网络的具体结构如下所示：
```
CNN1d(
  (conv1): Sequential(
    (0): Conv1d(1, 32, kernel_size=(5,), stride=(1,))
    (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,))
  )
  (conv2): Sequential(
    (0): Conv1d(32, 32, kernel_size=(3,), stride=(1,))
    (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (fc): Sequential(
    (0): Linear(in_features=3200, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=4, bias=True)
    (3): Softmax(dim=1)
  )
)
CNN2d(
  (conv1): Sequential(
    (0): Conv1d(5, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (1): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(1,))
    (2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv2): Sequential(
    (0): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (1): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(1,))
    (2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv3): Sequential(
    (0): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (1): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(1,))
    (2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv4): Sequential(
    (0): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (1): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(1,))
    (2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc): Sequential(
    (0): Linear(in_features=512, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=4, bias=True)
    (3): Softmax(dim=1)
  )
)

```

预处理操作主要体现在跨被试的条件下，我此次实验采用了三种方式来进行数据预处理，分别使：
* 每组数据不进行预处理，直接用作对照
* 每组数据分别进行归一化后再进行拼接
* 每组数据拼接好成为训练集和测试集后，对训练集进行归一化，并用此归一化条件来归一化测试集，这一种实验设定不需要事先获得测试集的所有数据。

我用被试依赖的实验条件来比较两种模型的优劣，因为跨被试实在跑的太慢了。用跨被试的实验条件来比较预处理不同所带来的性能差异

实验代码除了已经随附，也一并上传至 https://github.com/Otsuts/lbl


### 3.实验结果

#### 1. 被试依赖条件下的实验结果

| 所用模型  | 是否归一化 |  测试集准确率 |
| :------------- | :----------: | :------------:|
| 单通道卷积 |   否   |55.22|
| 多通道卷积 |   否     | 65.66|


![](pictures/picture2.png)


#### 2. 跨被试条件下的实验结果

| 归一化方式 | 所用模型 |  测试集准确率 |
| :------------- | :----------: | :------------: |
| 无归一化|   单通道卷积  | 95.59|
| 各数据分别归一化 |   单通道卷积     |   96.51  |
| 合并后归一化 |   单通道卷积    |     |

![](pictures/picture1.png)

#### 3. 实验结果分析

首先我必须感慨一下这次实验调试量之大，而且每个同学调出来结果都不一样，太吃模型架构了。其次我认识到了模型输入对于性能的影响，可以看到，如果不考虑脑电波数据的结构，直接按照SVM的方法，将其拼成一个一维向量，效果就不及将脑波的五个频段分别作为卷积神经网络的五个输入结果好。其次，模型的训练必须考虑到过拟合的因素，卷积神经网络的通道数不能过大，否则容易在测试集上过拟合，训练的轮数也不宜过多。

对于跨被试条件，我们可以看到对于数据进行归一化仍然对训练结果起着一定的促进作用，可以看到，相比于SVM， 卷积神经网络在数据量足够大的条件下，能够比较好的解决数据分布迁移的问题。而且性能相比传统的SVM有较大的提升。
