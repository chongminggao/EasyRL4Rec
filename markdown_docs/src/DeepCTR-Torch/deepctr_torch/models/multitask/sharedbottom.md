## ClassDef SharedBottom
Doc is waiting to be generated...
### FunctionDef __init__(self, dnn_feature_columns, bottom_dnn_hidden_units, tower_dnn_hidden_units, l2_reg_linear, l2_reg_embedding, l2_reg_dnn, init_std, seed, dnn_dropout, dnn_activation, dnn_use_bn, task_types, task_names, device, gpus)
**__init__**: 此函数的功能是初始化SharedBottom模型。

**参数**:
- **dnn_feature_columns**: DNN特征列，用于模型的输入特征。
- **bottom_dnn_hidden_units**: 底层DNN的隐藏单元，是一个元组。
- **tower_dnn_hidden_units**: 塔层DNN的隐藏单元，是一个元组。
- **l2_reg_linear**: 线性部分的L2正则化系数。
- **l2_reg_embedding**: 嵌入层的L2正则化系数。
- **l2_reg_dnn**: DNN部分的L2正则化系数。
- **init_std**: 权重初始化的标准差。
- **seed**: 随机种子。
- **dnn_dropout**: DNN层的dropout比率。
- **dnn_activation**: DNN层的激活函数。
- **dnn_use_bn**: 是否在DNN层使用批量归一化。
- **task_types**: 任务类型，是一个元组，例如('binary', 'binary')。
- **task_names**: 任务名称，是一个元组，例如('ctr', 'ctcvr')。
- **device**: 计算设备，例如'cpu'或'cuda'。
- **gpus**: 使用的GPU编号，None表示不使用GPU。

**代码描述**:
此函数首先调用父类的初始化方法，设置了线性特征列为空，DNN特征列为传入的dnn_feature_columns，以及其他相关的正则化、初始化参数。然后，它检查任务数量是否大于1，DNN特征列是否为空，以及任务类型的数量是否与任务名称的数量相等，确保输入的参数合法性。接着，对于每个任务类型，检查是否为合法的任务类型（'binary'或'regression'）。之后，计算输入特征的维度，初始化底层DNN和塔层DNN，并为每个任务添加最终的线性层和预测层。最后，将底层DNN、塔层DNN及其最终层的权重添加到正则化权重列表中，以便后续计算正则化损失，并将模型移动到指定的计算设备上。

此函数中使用了`DNN`类来构建底层DNN和塔层DNN，`PredictionLayer`类用于构建每个任务的预测层。通过`compute_input_dim`函数计算输入特征的维度，以及通过`add_regularization_weight`函数添加正则化权重。

**注意**:
- 确保传入的`dnn_feature_columns`非空，且包含了所有需要的特征信息。
- `task_types`和`task_names`的长度必须相等，且每个任务类型必须是支持的类型（'binary'或'regression'）。
- `bottom_dnn_hidden_units`和`tower_dnn_hidden_units`定义了底层DNN和塔层DNN的结构，需要根据具体任务调整这些参数以达到最佳性能。
- 使用`device`和`gpus`参数可以指定模型的计算设备，以适应不同的硬件环境。
***
### FunctionDef forward(self, X)
**forward**: 此函数的功能是执行模型的前向传播过程。

**参数**:
- **X**: 输入的特征数据，通常是一个Tensor，包含了模型需要处理的所有输入特征。

**代码描述**:
`forward` 函数是 `SharedBottom` 模型的核心执行函数，负责模型的前向传播过程。该函数首先通过调用 `input_from_feature_columns` 方法从输入特征中提取稀疏特征的嵌入表示和密集特征的值列表。接着，使用 `combined_dnn_input` 函数将稀疏特征的嵌入表示和密集特征的值合并为深度神经网络（DNN）的输入。然后，通过底层共享的DNN（即 `bottom_dnn`）处理这个合并后的输入，得到共享层的输出。

在共享层输出的基础上，针对每个任务，`forward` 函数分别通过特定的塔式DNN（`tower_dnn`）进行处理。如果定义了塔式DNN的隐藏层（`tower_dnn_hidden_units`），则先通过塔式DNN处理共享层的输出，再通过塔式DNN的最终层（`tower_dnn_final_layer`）获取逻辑输出；如果没有定义塔式DNN的隐藏层，则直接通过塔式DNN的最终层处理共享层的输出。最后，对每个任务的输出使用对应的输出层（`out`）进行处理，得到每个任务的最终输出，并将所有任务的输出在最后一个维度上进行拼接，形成最终的输出张量。

**注意**:
- 确保输入的特征数据 `X` 已经正确处理，包括必要的预处理和特征编码，以符合模型的输入要求。
- 该函数支持处理多任务学习场景，其中每个任务可以有自己的塔式DNN和输出层，但所有任务共享底层的DNN表示，这有助于学习跨任务的通用特征表示。

**输出示例**:
假设模型配置为处理两个任务，每个任务的输出维度为1，批量大小（batch size）为5。则 `forward` 函数的输出可能是一个形状为 `(5, 2)` 的Tensor，其中第一列代表第一个任务的输出，第二列代表第二个任务的输出。
***
