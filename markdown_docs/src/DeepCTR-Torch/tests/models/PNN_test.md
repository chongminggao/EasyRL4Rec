## FunctionDef test_PNN(use_inner, use_outter, kernel_type, sparse_feature_num)
**test_PNN**: 此函数用于测试PNN模型的性能。

**参数**:
- `use_inner`: 布尔值，指示是否在PNN模型中使用内积。
- `use_outter`: 布尔值，指示是否在PNN模型中使用外积。
- `kernel_type`: 字符串，指定PNN模型外积层使用的核类型，可以是`'mat'`、`'vec'`或`'num'`。
- `sparse_feature_num`: 整数，指定测试数据中稀疏特征的数量。

**代码描述**:
`test_PNN`函数首先通过`get_test_data`函数生成测试数据，包括模型输入、目标输出和特征列定义。这些测试数据根据提供的`sample_size`和`sparse_feature_num`参数动态生成，以模拟不同的测试场景。接着，函数实例化一个PNN模型，其中`dnn_hidden_units`、`dnn_dropout`、`use_inner`、`use_outter`和`kernel_type`参数用于定义模型的结构和行为。`get_device`函数被调用以确定模型运行的设备（CPU或GPU）。最后，`check_model`函数用于编译模型，执行训练和评估过程，并验证模型的保存和加载功能。

此函数通过不同的参数组合（如是否使用内积或外积，以及外积的核类型）来测试PNN模型的性能和功能，确保模型在各种配置下均能正常工作。这对于验证PNN模型的灵活性和鲁棒性至关重要。

**注意**:
- 在使用`test_PNN`函数进行模型测试时，应确保测试数据与模型预期的输入格式一致。
- `kernel_type`参数仅在`use_outter`为True时有效，这一点在设计测试用例时需要特别注意。
- 测试过程中，模型的训练和评估是自动进行的，但开发者应关注测试的输出，以便及时发现并解决潜在的问题。
