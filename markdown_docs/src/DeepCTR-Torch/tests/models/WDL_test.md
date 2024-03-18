## FunctionDef test_WDL(sparse_feature_num, dense_feature_num)
**test_WDL**: 该函数用于测试WDL（Wide & Deep Learning）模型的功能和性能。

**参数**:
- `sparse_feature_num`: 稀疏特征的数量。
- `dense_feature_num`: 密集特征的数量。

**代码描述**:
`test_WDL`函数首先定义了模型名称为"WDL"，并设置了样本大小为`SAMPLE_SIZE`。接着，通过调用`get_test_data`函数生成测试数据，包括模型输入`x`、目标输出`y`和特征列定义`feature_columns`。这些测试数据根据传入的稀疏特征数量和密集特征数量动态生成，以适应不同的测试场景。

然后，函数实例化了一个WDL模型，其中包括线性特征列和DNN特征列（在这里两者使用相同的`feature_columns`），并设置了DNN部分的激活函数为'prelu'、隐藏层单元为[32, 32]以及dropout比率为0.5。模型运行的设备通过调用`get_device`函数动态获取，以支持在CPU或GPU上运行。

最后，通过调用`check_model`函数，对WDL模型进行编译、训练和评估，并检查模型的保存和加载功能。这一步骤是确保模型能够正确训练，并且模型的输入/输出、保存/加载机制无误的关键。

**注意**:
- 在使用`test_WDL`函数进行模型测试时，需要确保传入的稀疏特征数量和密集特征数量与实际模型设计相匹配。
- 模型训练和评估的性能可能会受到所选设备（CPU或GPU）的影响，因此在进行性能敏感的测试时应考虑设备选择。
- `check_model`函数的使用需要注意保存和加载模型权重或整个模型需要足够的文件系统权限。
