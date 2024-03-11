## FunctionDef parse_args_processed
**parse_args_processed**: 该函数的功能是解析命令行参数，并返回解析后的参数对象。

**参数**: 该函数没有参数。

**代码描述**: `parse_args_processed` 函数使用 `argparse` 库创建一个解析器对象，用于解析命令行参数。在这个函数中，定义了一个名为 `--method` 的命令行参数，其类型为字符串（`str`），默认值为 `'maxmin'`。该参数用于指定数据预处理时使用的方法。函数通过调用 `parse_known_args` 方法来解析命令行参数，该方法返回一个包含两个元素的元组，其中第一个元素是一个包含所有已解析参数的命名空间对象。函数最终返回这个命名空间对象，其中包含了所有通过命令行指定的参数。

在项目中，`parse_args_processed` 函数被 `get_normalized_data` 函数调用。`get_normalized_data` 函数根据 `--method` 参数的值（`maxmin` 或其他值）来选择不同的数据规范化方法。如果 `--method` 的值为 `maxmin`，则使用最大最小规范化方法；否则，使用其他规范化方法。这表明 `parse_args_processed` 函数提供的参数直接影响数据预处理的行为和结果。

**注意**: 在使用 `parse_args_processed` 函数时，需要确保命令行参数的正确性和合理性，特别是 `--method` 参数的值，因为它直接影响到数据预处理的逻辑。

**输出示例**: 假设命令行中没有指定任何参数，那么函数的返回值可能如下所示：
```python
Namespace(method='maxmin')
```
这表示返回了一个命名空间对象，其中 `method` 参数的值为默认值 `'maxmin'`。
## FunctionDef get_normalized_data
**get_normalized_data**: 该函数的功能是对训练集和验证集中的视频时长和观看比例进行规范化处理。

**参数**: 该函数没有参数。

**代码描述**: `get_normalized_data` 函数首先通过调用 `parse_args_processed` 函数获取命令行参数，根据 `--method` 参数的值选择不同的数据规范化方法。如果 `--method` 的值为 `"maxmin"`，则对视频时长使用最大最小规范化方法；否则，对视频时长使用基于均值和标准差的规范化方法。接着，该函数计算视频的观看比例，并对其进行规范化处理，同样根据 `--method` 参数的值选择不同的规范化方法。最后，函数会将处理后的训练集和验证集数据保存到指定的文件路径，并返回这两个数据集。

在规范化过程中，对于观看比例的规范化，如果 `--method` 的值为 `"maxmin"`，则使用最大最小规范化方法；否则，使用基于均值和标准差的规范化方法，并对超出98百分位数的数据进行截断处理，以避免极端值的影响。此外，对于任何计算结果为无穷大的观看比例，将其值设置为1，对于缺失的规范化观看比例，将其值设置为0。

该函数还将视频ID列的名称从 `"video_id"` 改为 `"item_id"`，以符合后续处理的需要。处理完成后，训练集和验证集数据会分别保存到预先定义的文件路径中。

**注意**: 使用该函数前，需要确保已经通过 `parse_args_processed` 函数正确设置了命令行参数，特别是 `--method` 参数，因为它直接决定了数据规范化的方法。此外，还需要确保输入的训练集和验证集数据中包含 `"duration_ms"` 和 `"play_time_ms"` 两列，以便进行时长和观看比例的计算和规范化处理。

**输出示例**: 函数执行完成后，会返回两个数据集 `df_train` 和 `df_val`，它们分别包含了处理后的训练集和验证集数据。数据集中的 `"duration_normed"` 和 `"watch_ratio_normed"` 列分别表示规范化后的视频时长和观看比例。同时，函数会在控制台输出 `"saved_precessed_files!"`，表示处理后的文件已成功保存。
## FunctionDef get_df_data(filepath_input, usecols)
**get_df_data**: 该函数的功能是根据输入的文件路径和指定的列名，从处理后的训练集或测试集中读取数据。

**参数**:
- filepath_input: 输入文件的路径，该路径指向原始的训练集或测试集文件。
- usecols: 一个列表，指定需要从文件中读取的列名。如果为None，则读取所有列。

**代码描述**:
此函数首先通过`os.path.basename`获取输入文件路径的基础名称（即文件名），并断言这个文件名必须是"train_processed.csv"或"test_processed.csv"中的一个。这一步确保了函数仅处理这两个特定的文件。接着，根据文件名决定使用哪个预处理后的文件路径（`filepath_processed_train`或`filepath_processed_test`）。

如果预处理后的文件存在，则直接使用`pd.read_csv`读取该文件，并根据`usecols`参数决定是否只读取指定的列。如果文件不存在，则调用`get_normalized_data`函数获取规范化后的训练集和验证集数据。根据文件名，选择训练集或验证集作为`df_data`，并根据`usecols`参数决定是否只包含指定的列。

该函数与项目中的其他对象有以下调用关系：
- 被`KuaiRandData.py`中的`get_df`和`get_item_popularity`等方法调用，用于获取处理后的数据集，以便进行进一步的数据分析和模型训练。
- 调用`get_normalized_data`函数，该函数负责对数据进行规范化处理，包括视频时长和观看比例的规范化，以及将视频ID列的名称从"video_id"改为"item_id"，以符合后续处理的需要。

**注意**:
- 在使用`get_df_data`函数之前，需要确保已经有了正确处理并命名为"train_processed.csv"或"test_processed.csv"的文件，或者确保`get_normalized_data`函数能够生成这些文件。
- `get_normalized_data`函数的详细工作机制和参数设置对于正确使用`get_df_data`函数至关重要，因为它直接影响到`get_df_data`能否正确获取数据。

**输出示例**:
假设`usecols`参数设置为`['user_id', 'item_id', 'time_ms']`，则`get_df_data`函数可能返回如下格式的DataFrame：

| user_id | item_id | time_ms |
|---------|---------|---------|
| 1       | 101     | 158962  |
| 2       | 102     | 159883  |
| ...     | ...     | ...     |

此DataFrame包含了用户ID、项目ID和时间戳，具体列由`usecols`参数决定。
