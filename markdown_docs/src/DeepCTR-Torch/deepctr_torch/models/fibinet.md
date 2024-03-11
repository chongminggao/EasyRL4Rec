## ClassDef FiBiNET
**FiBiNET**: FiBiNET 是一个实现特征重要性和双线性特征交互网络架构的模型。

**属性**:
- `linear_feature_columns`: 用于模型线性部分的特征列。
- `dnn_feature_columns`: 用于模型深度部分的特征列。
- `bilinear_type`: 字符串，指定双线性交互层使用的双线性函数类型，可以是 `'all'`、`'each'` 或 `'interaction'`。
- `reduction_ratio`: 整数，SENET层使用的降维比率。
- `dnn_hidden_units`: 列表，指定DNN部分每层的单元数。
- `l2_reg_linear`: 浮点数，应用于模型宽部分的L2正则化强度。
- `l2_reg_embedding`: 浮点数，应用于嵌入向量的L2正则化强度。
- `l2_reg_dnn`: 浮点数，应用于DNN的L2正则化强度。
- `init_std`: 浮点数，用作嵌入向量的初始化标准差。
- `seed`: 整数，用作随机种子。
- `dnn_dropout`: 浮点数，DNN坐标丢弃的概率。
- `dnn_activation`: 激活函数，用于DNN。
- `task`: 字符串，指定任务类型，`"binary"` 表示二元逻辑回归，`"regression"` 表示回归损失。
- `device`: 字符串，指定模型运行的设备，`"cpu"` 或 `"cuda:0"`。
- `gpus`: 整数列表或torch.device，指定多GPU运行。如果为None，则在 `device` 上运行。

**代码描述**:
FiBiNET 类继承自 BaseModel 类，通过初始化参数构建一个具有特征重要性和双线性特征交互网络架构的模型。它首先通过基类初始化模型的基础结构，包括特征嵌入、线性模型部分和预测层。然后，FiBiNET 特有的结构包括 SENET 层和双线性交互层，用于提取和处理特征间的交互信息。DNN 部分用于进一步提取特征的深层表示。模型的输出是通过线性和DNN部分的结果相加得到的，适用于二分类或回归任务。

在项目中，FiBiNET 通过 `FiBiNET_test.py` 中的 `test_FiBiNET` 函数进行测试，该测试函数验证了 FiBiNET 模型在不同双线性类型下的性能。

**注意**:
- 在使用 FiBiNET 模型时，需要确保输入的特征列与模型预期的一致。
- 模型训练前，需要调用 compile 方法来配置优化器、损失函数等训练参数。
- 由于模型支持在 GPU 上运行，因此在有多个 GPU 可用的情况下，可以通过 `gpus` 参数指定使用的 GPU。

**输出示例**:
FiBiNET 模型的输出是一个预测值，对于二分类任务，输出值在 0 到 1 之间；对于回归任务，输出值为一个连续值。例如，在二分类任务中，模型可能会输出每个样本属于正类的概率。
### FunctionDef __init__(self, linear_feature_columns, dnn_feature_columns, bilinear_type, reduction_ratio, dnn_hidden_units, l2_reg_linear, l2_reg_embedding, l2_reg_dnn, init_std, seed, dnn_dropout, dnn_activation, task, device, gpus)
**__init__**: FiBiNET模型的初始化函数。

**参数**:
- **linear_feature_columns**: 线性特征列，用于模型的线性部分。
- **dnn_feature_columns**: DNN特征列，用于模型的深度神经网络部分。
- **bilinear_type**: 双线性交互类型，默认为'interaction'。
- **reduction_ratio**: SENET层的降维比率，默认为3。
- **dnn_hidden_units**: DNN层的隐藏单元数，默认为(128, 128)。
- **l2_reg_linear**: 线性部分的L2正则化系数，默认为1e-5。
- **l2_reg_embedding**: 嵌入层的L2正则化系数，默认为1e-5。
- **l2_reg_dnn**: DNN部分的L2正则化系数，默认为0。
- **init_std**: 权重初始化的标准差，默认为0.0001。
- **seed**: 随机种子，默认为1024。
- **dnn_dropout**: DNN层的dropout率，默认为0。
- **dnn_activation**: DNN层的激活函数，默认为'relu'。
- **task**: 任务类型，默认为'binary'。
- **device**: 运行设备，默认为'cpu'。
- **gpus**: GPU设备列表，默认为None。

**代码描述**:
FiBiNET模型的初始化函数首先调用基类的初始化方法，传入线性特征列、DNN特征列以及其他相关参数。然后，它初始化了几个关键的网络层，包括SENETLayer、BilinearInteraction和DNN。SENETLayer用于实现特征重要性的自适应学习，BilinearInteraction用于实现特征间的双线性交互，DNN用于深度特征学习。此外，还有一个线性层dnn_linear，用于从DNN的输出中预测最终结果。

SENETLayer的初始化依赖于字段大小（即embedding_dict的长度），降维比率、随机种子和运行设备。BilinearInteraction的初始化依赖于字段大小、嵌入维度、双线性类型、随机种子和运行设备。DNN的初始化依赖于DNN特征列的输入维度计算结果、隐藏单元数、激活函数、L2正则化系数、dropout率、权重初始化标准差和运行设备。

**注意**:
- 在使用FiBiNET模型时，需要确保传入的特征列正确无误，包括线性特征列和DNN特征列。
- bilinear_type参数决定了特征间双线性交互的类型，其值应为'all'、'each'或'interaction'中的一个。
- reduction_ratio参数影响SENETLayer的降维效果，需要根据具体情况调整。
- dnn_hidden_units、dnn_dropout和dnn_activation参数影响DNN层的结构和性能，应根据任务需求进行配置。
- task参数指定了任务类型，通常为'binary'（二分类任务）或'multiclass'（多分类任务）。
- device和gpus参数用于指定模型的运行设备，根据实际运行环境进行设置。
***
### FunctionDef compute_input_dim(self, feature_columns, include_sparse, include_dense)
**compute_input_dim**: 此函数的功能是计算模型输入的维度。

**参数**:
- `feature_columns`: 特征列，包含了模型中所有的特征信息。
- `include_sparse`: 布尔值，指示是否包含稀疏特征的维度计算，默认为True。
- `include_dense`: 布尔值，指示是否包含密集特征的维度计算，默认为True。

**代码描述**:
`compute_input_dim`函数主要用于计算FiBiNET模型输入的维度。它首先根据传入的`feature_columns`参数，将特征列分为稀疏特征列和密集特征列两部分。稀疏特征列包括`SparseFeat`和`VarLenSparseFeat`类型的特征，而密集特征列则包括`DenseFeat`类型的特征。

对于稀疏特征，函数计算字段大小（即稀疏特征的数量），并根据第一个稀疏特征的嵌入维度以及字段大小计算稀疏输入的维度。对于密集特征，函数简单地将所有密集特征的维度相加以得到密集输入的维度。

根据`include_sparse`和`include_dense`参数的值，函数决定是否将稀疏输入维度和/或密集输入维度包含在最终的输入维度计算中。最后，函数返回计算得到的输入维度。

在FiBiNET模型初始化过程中，`compute_input_dim`函数被调用以确定DNN层的输入大小。这是因为FiBiNET模型需要根据输入特征的维度来构建其深度神经网络部分，确保模型的输入层与实际输入数据的维度相匹配。

**注意**:
- 确保传入的`feature_columns`参数正确无误，包含了模型所需的所有特征信息。
- 当特征列中不包含稀疏特征时，`include_sparse`参数的设置将不会影响最终的输入维度计算结果；同理，当不包含密集特征时，`include_dense`参数的设置也不会影响结果。

**输出示例**:
假设模型中有3个稀疏特征，每个特征的嵌入维度为4，以及2个密集特征，每个特征的维度为1。如果`include_sparse`和`include_dense`参数均设置为True，则`compute_input_dim`函数的返回值将是：
```
稀疏输入维度 = 3 * (3 - 1) * 4 = 24
密集输入维度 = 2 * 1 = 2
总输入维度 = 24 + 2 = 26
```
因此，函数将返回26作为模型输入的维度。
***
### FunctionDef forward(self, X)
**forward**: 此函数的功能是执行FiBiNET模型的前向传播过程。

**参数**:
- **X**: 输入的特征数据，通常是一个Tensor，包含了模型需要的所有特征信息。

**代码描述**:
`forward` 函数首先通过调用 `input_from_feature_columns` 方法从输入特征中提取稀疏特征的嵌入表示和密集特征的值列表。这一步骤是将原始输入特征转换为模型可以处理的形式的关键环节。

接着，将所有稀疏特征的嵌入表示通过 `torch.cat` 方法在第二维度上进行拼接，形成一个统一的稀疏特征嵌入输入。

然后，模型分别对拼接后的稀疏特征嵌入输入进行SE（Squeeze-and-Excitation）网络和双线性交互操作，得到SE网络输出和双线性交互的输出。SE网络通过对特征重要性进行建模，增强了模型对重要特征的关注能力，而双线性交互则用于捕获特征间的复杂交互关系。

此外，模型还对原始的稀疏特征嵌入输入直接进行双线性交互操作，以捕获不同特征组合间的交互信息。

模型同时计算线性部分的逻辑回归输出和通过DNN网络的输出。线性部分直接对输入特征进行线性变换，而DNN部分则通过深度神经网络学习特征的非线性组合和交互。

根据模型配置，最终的模型输出可以是线性部分和DNN部分的输出之和，或者是其中之一，这取决于模型是否同时使用线性特征列和DNN特征列。

最后，模型通过一个输出层将最终的逻辑回归结果转换为预测结果。

**注意**:
- 输入的特征数据X应包含模型所需的所有特征信息，包括稀疏特征和密集特征。
- 确保在调用此函数前，所有特征列的定义和嵌入字典已正确设置。

**输出示例**:
假设模型的输出层是一个sigmoid函数，用于二分类任务。那么，调用`forward`函数后，可能得到的输出是一个形状为`(batch_size, 1)`的Tensor，其中每个元素的值介于0和1之间，表示每个样本属于正类的概率。
***
