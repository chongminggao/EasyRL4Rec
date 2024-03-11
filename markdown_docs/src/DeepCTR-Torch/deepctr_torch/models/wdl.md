## ClassDef WDL
**WDL**: WDL 类实现了 Wide & Deep Learning 架构。

**属性**:
- `linear_feature_columns`: 用于模型线性部分的特征列。
- `dnn_feature_columns`: 用于模型深度部分的特征列。
- `dnn_hidden_units`: DNN部分的隐藏层单元数和层数。
- `l2_reg_linear`: 线性部分的L2正则化强度。
- `l2_reg_embedding`: 嵌入向量的L2正则化强度。
- `l2_reg_dnn`: DNN部分的L2正则化强度。
- `init_std`: 嵌入向量初始化的标准差。
- `seed`: 随机种子。
- `dnn_dropout`: DNN部分的dropout比率。
- `dnn_activation`: DNN部分的激活函数。
- `task`: 模型任务类型，"binary"表示二分类，"regression"表示回归。
- `device`: 模型运行的设备，"cpu"或"cuda:0"。
- `gpus`: 用于训练的GPU列表，如果为None，则运行在`device`指定的设备上。

**代码描述**:
WDL 类继承自 BaseModel 类，实现了 Wide & Deep Learning 架构。在初始化过程中，WDL 类首先调用基类的初始化方法，设置了线性特征列、DNN特征列、正则化参数等。然后，根据DNN特征列和隐藏层单元数的配置，决定是否使用DNN部分。如果使用DNN，会构建一个DNN网络，并将其输出连接到一个线性层以产生最终的预测。此外，WDL 类还包含了正则化权重的添加和模型的前向传播逻辑。

在项目中，WDL 类通过 src/DeepCTR-Torch/tests/models/WDL_test.py 中的测试函数 test_WDL 被调用，用于验证模型的正确性和性能。测试函数中通过传入不同的特征列配置和DNN参数，创建WDL模型实例，并对其进行测试。

**注意**:
- 在使用WDL模型时，需要确保传入的特征列与模型预期的一致。
- 模型的训练和评估需要在指定的设备上进行，可以是CPU或GPU。
- 正则化参数、初始化标准差、随机种子等参数的选择会影响模型的训练效果和性能。

**输出示例**:
假设模型被用于二分类任务，对于给定的输入特征，模型的输出可能是一个介于0和1之间的概率值，表示样本属于正类的概率。例如，对于某个样本，模型的输出可能是0.85，表示该样本有85%的概率属于正类。
### FunctionDef __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units, l2_reg_linear, l2_reg_embedding, l2_reg_dnn, init_std, seed, dnn_dropout, dnn_activation, dnn_use_bn, task, device, gpus)
**__init__**: 此函数用于初始化WDL模型对象。

**参数**:
- **linear_feature_columns**: 线性部分的特征列。
- **dnn_feature_columns**: DNN部分的特征列。
- **dnn_hidden_units**: DNN隐藏层的单元数，默认为(256, 128)。
- **l2_reg_linear**: 线性部分的L2正则化系数，默认为1e-5。
- **l2_reg_embedding**: 嵌入层的L2正则化系数，默认为1e-5。
- **l2_reg_dnn**: DNN部分的L2正则化系数，默认为0。
- **init_std**: 权重初始化的标准差，默认为0.0001。
- **seed**: 随机种子，默认为1024。
- **dnn_dropout**: DNN层的dropout率，默认为0。
- **dnn_activation**: DNN层的激活函数，默认为'relu'。
- **dnn_use_bn**: 是否在DNN层使用批量归一化，默认为False。
- **task**: 任务类型，默认为'binary'。
- **device**: 计算设备，默认为'cpu'。
- **gpus**: 使用的GPU，默认为None。

**代码描述**:
此函数首先调用基类的初始化方法，传入线性特征列、DNN特征列、L2正则化系数、初始化标准差、随机种子、任务类型、计算设备和GPU信息。接着，根据DNN特征列和DNN隐藏层单元数的长度判断是否使用DNN。如果使用DNN，将构建一个DNN对象，并创建一个线性层用于DNN的输出。此外，还会为DNN层和线性层的权重添加L2正则化。最后，将模型移动到指定的计算设备上。

此函数中调用了`DNN`类来构建深度神经网络，用于处理DNN特征列。通过`compute_input_dim`方法计算DNN输入层的维度，确保网络能够接收正确形状的输入数据。同时，使用`add_regularization_weight`方法为DNN层和线性层的权重添加L2正则化，有助于控制模型的过拟合问题。

**注意**:
- 确保传入的特征列参数正确无误，包含了所有模型所需的特征信息。
- DNN隐藏层单元数和激活函数的选择对模型性能有重要影响，应根据具体任务进行调整。
- L2正则化系数的设置对于防止模型过拟合非常关键，需要根据实际情况进行合理设置。
- 计算设备的选择（CPU或GPU）将直接影响模型的训练速度和效率，根据实际硬件环境进行选择。
***
### FunctionDef forward(self, X)
**forward**: 此函数的功能是执行模型的前向传播过程。

**参数**:
- **X**: 输入的特征数据，通常是一个Tensor。

**代码描述**:
`forward` 函数首先调用 `input_from_feature_columns` 方法从输入的特征数据 `X` 中提取出稀疏特征的嵌入表示列表和密集特征的值列表。这一步骤是通过分析输入数据与模型定义的特征列之间的关系来完成的，确保了不同类型的输入特征能够被正确处理并转换为模型可以直接使用的格式。

接着，函数调用 `linear_model` 方法计算线性部分的逻辑值（logit）。这一步涉及到将输入特征通过一个线性层（或多个线性层）进行处理，以生成一个预测值。

如果模型配置为使用深度神经网络（DNN），则会进一步处理稀疏特征的嵌入表示和密集特征的值。这一处理是通过 `combined_dnn_input` 函数完成的，该函数将稀疏特征的嵌入表示和密集特征的值合并为一个张量，以供DNN使用。之后，通过DNN层处理这个合并后的张量，得到DNN的输出，并通过一个线性层（`dnn_linear`）计算DNN部分的逻辑值。最后，将线性部分的逻辑值与DNN部分的逻辑值相加，得到最终的逻辑值。

最后，通过 `out` 方法将最终的逻辑值转换为预测值 `y_pred`，并返回这个预测值。

**注意**:
- 确保输入的特征数据 `X` 与模型定义的特征列相匹配，包括特征的类型和维度。
- 如果模型配置为不使用DNN，则不会执行与DNN相关的处理步骤。

**输出示例**:
假设模型的输出层是一个sigmoid激活函数，那么 `forward` 函数的返回值 `y_pred` 将是一个形状为 `(batch_size, 1)` 的Tensor，其中的每个元素都是在0到1之间，表示每个样本的预测概率。
***
