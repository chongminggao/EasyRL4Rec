## ClassDef PLE
Doc is waiting to be generated...
### FunctionDef __init__(self, dnn_feature_columns, shared_expert_num, specific_expert_num, num_levels, expert_dnn_hidden_units, gate_dnn_hidden_units, tower_dnn_hidden_units, l2_reg_linear, l2_reg_embedding, l2_reg_dnn, init_std, seed, dnn_dropout, dnn_activation, dnn_use_bn, task_types, task_names, device, gpus)
**__init__**: 此函数用于初始化PLE（Progressive Layered Extraction）模型对象。

**参数**:
- **dnn_feature_columns**: 特征列，用于定义模型的输入特征。
- **shared_expert_num**: 共享专家网络的数量，默认为1。
- **specific_expert_num**: 特定任务专家网络的数量，默认为1。
- **num_levels**: PLE模型的层数，默认为2。
- **expert_dnn_hidden_units**: 专家网络的隐藏单元，默认为(256, 128)。
- **gate_dnn_hidden_units**: 门控网络的隐藏单元，默认为(64,)。
- **tower_dnn_hidden_units**: 塔网络的隐藏单元，默认为(64,)。
- **l2_reg_linear**: 线性部分的L2正则化系数，默认为0.00001。
- **l2_reg_embedding**: 嵌入部分的L2正则化系数，默认为0.00001。
- **l2_reg_dnn**: DNN部分的L2正则化系数，默认为0。
- **init_std**: 权重初始化的标准差，默认为0.0001。
- **seed**: 随机种子，默认为1024。
- **dnn_dropout**: DNN层的dropout比率，默认为0。
- **dnn_activation**: DNN层的激活函数，默认为'relu'。
- **dnn_use_bn**: 是否在DNN层使用批量归一化，默认为False。
- **task_types**: 任务类型列表，默认为('binary', 'binary')。
- **task_names**: 任务名称列表，默认为('ctr', 'ctcvr')。
- **device**: 计算设备，默认为'cpu'。
- **gpus**: 使用的GPU列表，默认为None。

**代码描述**:
此函数首先调用父类的初始化方法，设置了模型的基本参数。接着，它进行了一系列的参数校验，包括任务数量、特征列的非空校验、任务类型的合法性校验等。之后，根据输入参数初始化了特定任务专家网络、共享专家网络、门控网络和塔网络的结构。这些网络结构的初始化依赖于`DNN`类来构建深度神经网络层，并使用`PredictionLayer`类来处理最终的预测输出。此外，还涉及到了`compute_input_dim`方法来计算输入特征的维度，以及`add_regularization_weight`方法来添加正则化权重，这些都是为了构建和优化PLE模型的关键步骤。

**注意**:
- 确保传入的`task_types`和`task_names`长度一致，且`task_types`中的任务类型为支持的类型（'binary'或'regression'）。
- `dnn_feature_columns`不能为空，因为它定义了模型的输入特征。
- 使用时应注意`device`和`gpus`参数的设置，以确保模型能够在正确的计算设备上运行。

**输出示例**:
由于`__init__`函数是用于初始化PLE模型的，因此它不直接产生输出。但初始化后的PLE模型对象将包含多个层级的专家网络、门控网络和塔网络，这些网络结构和参数将用于后续的训练和预测过程。
#### FunctionDef multi_module_list(num_level, num_tasks, expert_num, inputs_dim_level0, inputs_dim_not_level0, hidden_units)
**multi_module_list**: 此函数用于创建一个多层次、多任务的专家网络模块列表。

**参数**:
- **num_level**: 整数，表示网络的层次数。
- **num_tasks**: 整数，表示任务的数量。
- **expert_num**: 整数，表示每个任务的专家数量。
- **inputs_dim_level0**: 整数，表示第0层的输入维度。
- **inputs_dim_not_level0**: 整数，表示非第0层的输入维度。
- **hidden_units**: 列表，包含每个隐藏层的单元数。

**代码描述**:
`multi_module_list`函数通过嵌套的方式创建了一个三维的`nn.ModuleList`结构，用于构建多层次、多任务的专家网络。这个结构的每一层对应网络的一个层次，每个任务在每个层次中都有多个专家（DNN模型）。函数首先遍历每个层次（`num_level`），对于每个层次，它创建一个任务级的`nn.ModuleList`。在每个任务级别，它又创建了一个专家级的`nn.ModuleList`，其中包含了`expert_num`个DNN模型。DNN模型的输入维度根据层次而变化：第0层使用`inputs_dim_level0`，其他层使用`inputs_dim_not_level0`。DNN模型的其他参数（如激活函数、L2正则化强度、Dropout率等）在此函数中未直接指定，需在调用此函数时通过外部变量传入。

此函数利用了DNN类（定义在`src/DeepCTR-Torch/deepctr_torch/layers/core.py`中）来构建每个专家网络。DNN类是一个深度神经网络模块，用于学习输入特征的深度表示。在此上下文中，DNN模型作为专家网络，用于处理特定任务的特征学习。

**注意**:
- 确保`num_level`、`num_tasks`和`expert_num`均为正整数，且`hidden_units`为非空列表，否则可能无法正确构建网络结构。
- 此函数创建的网络结构复杂度较高，需要合理配置硬件资源以支持大规模的参数训练。
- 在实际应用中，需要根据具体任务调整DNN模型的参数（如`hidden_units`、激活函数等），以达到最佳性能。

**输出示例**:
假设调用`multi_module_list(2, 3, 4, 10, 20, [64, 32])`，将返回一个包含2个层次的`nn.ModuleList`，每个层次包含3个任务，每个任务包含4个DNN模型。第0层的DNN模型的输入维度为10，其他层的DNN模型的输入维度为20，每个DNN模型的隐藏层单元数为[64, 32]。
***
***
### FunctionDef cgc_net(self, inputs, level_num)
**cgc_net**: 该函数用于实现基于任务的专家网络和共享专家网络的构建，并通过门控机制动态地为每个任务选择合适的专家网络输出。

**参数**:
- inputs: 一个包含各个任务输入以及共享任务输入的列表。
- level_num: 当前PLE网络的层级编号。

**代码描述**:
`cgc_net` 函数首先定义了两种类型的专家网络：针对特定任务的专家网络和共享的专家网络。对于每个任务，它会分别通过这些专家网络处理输入，然后使用门控机制来动态选择哪些专家的输出最适合当前任务。这个过程分为以下几个步骤：

1. **专家网络构建**：函数开始时，会为每个任务构建特定的专家网络，并为所有任务共享的输入构建共享专家网络。这些网络的输出被收集起来，以便后续处理。

2. **门控机制**：接下来，对于每个任务，函数会将该任务的特定专家网络输出和共享专家网络输出合并，并通过一个门控网络来决定各个专家输出的重要性。门控网络的输出用于加权合并专家网络的输出，得到最终的任务特定输出。

3. **共享专家门控**：除了为每个特定任务设计的门控机制外，还有一个针对共享专家网络输出的门控机制，它同样通过门控网络来加权合并所有专家的输出。

在`PLE`模型的`forward`方法中，`cgc_net`函数被用于处理多任务学习中的每一层，通过不断迭代，每一层的输出都会作为下一层的输入，从而实现复杂任务间的知识共享和转移。

**注意**:
- 该函数依赖于PyTorch框架，因此输入和输出都是PyTorch张量（Tensor）。
- 使用该函数之前，需要确保已经正确初始化了专家网络和门控网络的参数。
- 门控网络的设计和参数设置对模型性能有重要影响，需要根据具体任务进行调整。

**输出示例**:
假设有两个任务，每个任务有2个专家网络，共享专家网络数量为1，那么`cgc_net`函数的输出可能是一个包含3个元素的列表，每个元素对应一个任务的加权专家网络输出，以及共享任务的输出。每个输出是一个维度为`(batch_size, feature_dim)`的张量，其中`batch_size`是批处理大小，`feature_dim`是特征维度。
***
### FunctionDef forward(self, X)
**forward**: 此函数的功能是执行PLE模型的前向传播过程。

**参数**:
- **X**: 输入的特征数据，通常是一个Tensor。

**代码描述**:
`forward` 函数首先通过调用`input_from_feature_columns`函数从输入特征中提取稀疏和密集特征的嵌入表示和值列表。接着，使用`combined_dnn_input`函数将这些特征合并为深度神经网络(DNN)的输入。

在准备好DNN的输入后，函数将这个输入复制多次，以生成CGC（Cross-Gate Component）网络的输入。这里的复制次数等于任务数加一，意味着每个任务有一个输入，外加一个共享的输入。

接下来，`forward` 函数通过多层循环，使用`cgc_net`函数处理CGC网络的每一层。在每一层中，`cgc_net`函数负责执行基于任务的专家网络和共享专家网络的构建，并通过门控机制动态地为每个任务选择合适的专家网络输出。每一层的输出都会作为下一层的输入。

最后，对于每个任务，函数使用特定的塔式DNN（如果有的话）处理`cgc_net`的输出，并通过最终层生成任务的输出。所有任务的输出最终被拼接在一起，形成模型的最终输出。

**注意**:
- 确保输入的特征数据`X`已经正确地转换为模型可以处理的格式。
- 该函数依赖于`input_from_feature_columns`和`combined_dnn_input`函数来处理输入特征，以及`cgc_net`函数来执行CGC网络的前向传播。因此，需要确保这些函数已经正确实现并可以被调用。
- `forward`函数的输出依赖于模型的任务数和每个任务的塔式DNN配置，因此在使用前需要正确设置这些参数。

**输出示例**:
假设模型配置为处理两个任务，并且每个任务的塔式DNN输出维度为1。那么，`forward`函数的输出可能是一个形状为`(batch_size, 2)`的张量，其中`batch_size`是输入特征数据的批处理大小，2代表两个任务的输出。
***
