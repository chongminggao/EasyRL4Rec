## FunctionDef parse_args_processed
**parse_args_processed**: 此函数的功能是解析命令行参数，并返回解析后的参数对象。

**参数**：此函数没有参数。

**代码描述**：
`parse_args_processed` 函数使用 `argparse` 库创建一个解析器对象，用于解析命令行参数。在这个函数中，定义了一个名为 `--method` 的命令行参数，其类型为字符串（`str`），默认值为 `'maxmin'`。这意味着，如果在命令行中不指定 `--method` 参数，它将默认使用 `'maxmin'` 方法。然后，函数通过调用 `parse_known_args` 方法来解析命令行参数，该方法返回一个包含两个元素的元组，其中第一个元素是一个包含所有已解析参数的命名空间对象。函数最终返回这个命名空间对象。

在项目中，`parse_args_processed` 函数被 `get_normalized_data` 函数调用。`get_normalized_data` 函数利用 `parse_args_processed` 返回的参数对象来决定数据标准化的方法。如果 `--method` 参数的值为 `'maxmin'`，则使用最大最小值归一化方法；否则，使用其他方法（如高斯归一化）。这表明 `parse_args_processed` 函数在数据预处理流程中起着配置数据标准化方法的作用。

**注意**：在使用此函数时，需要确保 `argparse` 库已被正确导入。此外，虽然当前代码中只定义了一个 `--method` 参数，但根据实际需求，开发者可以通过添加更多的 `add_argument` 调用来扩展解析器的功能。

**输出示例**：
假设命令行中没有指定任何参数，函数的返回值可能如下所示：
```python
Namespace(method='maxmin')
```
这表示返回了一个命名空间对象，其中 `method` 参数的值为 `'maxmin'`。如果在命令行中指定了 `--method gaussian`，则返回值将为：
```python
Namespace(method='gaussian')
```
## FunctionDef get_normalized_data
**get_normalized_data**: 此函数的功能是对视频数据进行归一化处理。

**参数**: 此函数不接受任何直接参数。

**代码描述**:
`get_normalized_data` 函数首先通过调用 `parse_args_processed` 函数获取命令行参数，决定使用哪种归一化方法。支持的方法包括最大最小值归一化（"maxmin"）和基于均值和标准差的归一化。

- 当选择最大最小值归一化时，函数会计算大数据集（df_big）和小数据集（df_small）中视频时长（video_duration）的最大值和最小值，然后根据这些值对两个数据集中的视频时长进行归一化处理。
- 若选择的是基于均值和标准差的归一化方法，则函数会计算所有视频的平均时长和标准差，并使用这些统计信息来归一化大数据集和小数据集中的视频时长。

接下来，函数对视频的观看比例（watch_ratio）进行归一化处理，处理方法与视频时长的归一化相似，也是根据命令行参数选择最大最小值归一化或基于均值和标准差的归一化。此外，对于基于均值和标准差的归一化方法，函数还会将观看比例的归一化值中超过98百分位数的值设为98百分位数的值，以限制极端值的影响。

归一化处理完成后，函数会将大数据集和小数据集中的"video_id"列重命名为"item_id"，并将处理后的数据集分别保存到指定的文件路径。最后，函数打印出"saved_precessed_files!"，表示文件已成功保存，并返回处理后的大数据集和小数据集。

**注意**:
- 在使用此函数之前，需要确保已经通过 `parse_args_processed` 函数设置了正确的归一化方法。
- 此函数依赖于全局变量（如df_big, df_small, filepath_processed_big, filepath_processed_small等）和外部库（如pandas, numpy等），请确保在调用此函数前这些依赖已正确设置和导入。

**输出示例**:
由于此函数的输出依赖于输入数据和选择的归一化方法，因此无法提供一个固定的输出示例。但一般情况下，函数会返回两个经过归一化处理的pandas DataFrame对象，分别对应处理后的大数据集和小数据集。
## FunctionDef get_df_data(filepath_input, usecols)
**get_df_data**: 此函数的功能是从指定的文件路径中读取数据，并根据需要选择性地加载特定的列。

**参数**:
- filepath_input: 字符串类型，指定要读取数据的文件路径。
- usecols: 列表类型，可选参数，指定需要从文件中加载的列名。

**代码描述**:
`get_df_data` 函数首先根据输入的文件路径提取文件名，然后断言文件名必须是"big_matrix_processed.csv"或"small_matrix_processed.csv"中的一个。接着，根据文件名决定使用大数据集还是小数据集的处理后文件路径。如果该路径下的文件存在，则直接使用pandas的read_csv函数读取数据，可以通过usecols参数指定需要加载的列。如果文件不存在，则调用`get_normalized_data`函数获取归一化处理后的大数据集或小数据集。如果指定了usecols，则从这些数据集中选择指定的列，否则返回整个数据集。

此函数与项目中其他部分的关系主要体现在它作为数据预处理的一环，为后续的数据分析和模型训练提供准备好的数据集。它被`KuaiData.py`中的多个方法调用，例如`get_df`、`get_item_popularity`、`get_lbe`、`load_mat`和`load_video_duration`，这些方法依赖于`get_df_data`来加载和准备数据，以进行进一步的处理和分析。

**注意**:
- 在调用此函数之前，需要确保文件路径正确，且文件名符合要求。
- 此函数依赖于`get_normalized_data`函数来处理数据不存在时的情况，因此在使用之前需要确保相关的归一化处理逻辑已正确实现。
- 函数的性能和结果可能受到输入文件大小和指定列的影响，建议仅加载需要的列以优化性能。

**输出示例**:
由于此函数的输出依赖于输入文件和选择的列，因此无法提供一个固定的输出示例。但一般情况下，函数会返回一个pandas DataFrame对象，其中包含了从指定文件中加载的数据。如果指定了usecols参数，则DataFrame仅包含这些列，否则包含文件中的所有列。
