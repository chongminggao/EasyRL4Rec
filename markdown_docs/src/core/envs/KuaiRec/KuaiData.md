## ClassDef KuaiData
**KuaiData**: KuaiData 类用于处理和提供快手推荐系统相关的数据。

**属性**:
- train_data_path: 存储训练数据路径的字符串，指向"big_matrix_processed.csv"。
- val_data_path: 存储验证数据路径的字符串，指向"small_matrix_processed.csv"。

**代码描述**:
KuaiData 类继承自 BaseData 类，专门用于处理快手推荐系统的数据。它重写了 BaseData 类的一些方法，并添加了一些特有的方法来满足快手推荐系统数据处理的需求。

- `__init__` 方法初始化数据路径。
- `get_features` 方法返回用户特征、物品特征和奖励特征的列表。
- `get_df` 方法根据给定的文件名加载数据，并进行一系列的数据处理操作，如连接用户特征和物品特征，返回处理后的数据框架。
- `get_domination` 方法计算并返回物品特征的统治性数据。
- `get_item_similarity` 方法计算并返回物品之间的相似度。
- `get_item_popularity` 方法计算并返回物品的流行度。
- `load_user_feat` 和 `load_item_feat` 方法分别加载用户特征和物品特征。
- `get_lbe` 方法用于获取小数据集中用户和物品的标签编码器。
- 类中还包含几个静态方法，如 `load_mat`、`load_category`、`load_video_duration`、`get_similarity_mat` 和 `get_saved_distance_mat`，这些方法用于加载和处理数据，计算相似度和距离矩阵。

KuaiData 类与项目中的 KuaiEnv 类紧密相关，KuaiEnv 类通过调用 KuaiData 类的方法来加载和处理环境数据，以便在快手推荐系统环境中进行模拟和训练。

**注意**:
- 在使用 KuaiData 类时，需要确保 DATAPATH 和 PRODATAPATH 路径正确设置，这两个路径分别用于指向原始数据和处理后数据的存储位置。
- 在处理大规模数据时，部分方法可能需要较长的执行时间，特别是在首次计算物品相似度和流行度时，因为这些操作涉及到大量的数据处理和计算。

**输出示例**:
由于 KuaiData 类主要用于数据处理，其输出通常是数据框架（如 pandas DataFrame）或特定格式的数据（如 numpy 数组）。例如，`get_df` 方法可能返回如下格式的 DataFrame：

```
   user_id  item_id  timestamp  watch_ratio_normed  duration_normed  user_feature1  item_feature1
0       1       23   20230101               0.95              1.20              3              5
1       2       45   20230102               0.80              1.00              2              4
...
```

这个 DataFrame 包含了用户ID、物品ID、时间戳、标准化的观看比例、标准化的持续时间以及用户和物品的特征。
### FunctionDef __init__(self)
**__init__**: 此函数用于初始化KuaiData类的实例。

**参数**: 此函数不接受任何外部参数。

**代码描述**: 在KuaiData类的实例被创建时，`__init__`函数会被自动调用。此函数首先通过`super(KuaiData, self).__init__()`调用其父类的构造函数，确保父类被正确初始化。接着，函数内部初始化了两个实例变量：`train_data_path`和`val_data_path`。这两个变量分别被赋予了字符串值"big_matrix_processed.csv"和"small_matrix_processed.csv"，它们代表了训练数据和验证数据文件的路径。这意味着KuaiData类的实例将会使用这两个文件路径来访问或处理相应的数据集。

**注意**: 
- 确保在使用KuaiData类之前，相应的数据文件"big_matrix_processed.csv"和"small_matrix_processed.csv"已经存在于预定的路径下。这是因为类的实例会依赖这些路径来进行数据的读取和处理。
- 此函数不接受任何参数，因此在创建KuaiData类的实例时，不需要传递任何额外的信息。
- 由于此函数主要负责初始化操作，因此在实例化KuaiData类之后，通常不需要直接调用此函数。
***
### FunctionDef get_features(self, is_userinfo)
**get_features**: 此函数的功能是获取用户特征、项目特征和奖励特征的列表。

**参数**:
- is_userinfo: 可选参数，用于指定是否需要用户信息来获取特征，其具体用途在代码中未明确指定。

**代码描述**:
`get_features` 函数定义在 `KuaiData` 类中，旨在提供用于推荐系统的特征列表。函数根据是否需要用户信息（`is_userinfo` 参数），返回三个列表：用户特征（`user_features`）、项目特征（`item_features`）和奖励特征（`reward_features`）。用户特征仅包含一个元素 `"user_id"`，项目特征包含 `"item_id"` 和四个形如 `"featX"` 的特征（其中 `X` 是从 0 到 3 的数字），以及 `"duration_normed"`。奖励特征包含单一元素 `"watch_ratio_normed"`。

在项目中，`get_features` 函数被 `EnsembleModel` 类的 `save_all_models` 方法调用。在该方法中，首先通过调用 `get_features` 获取用户特征、项目特征和奖励特征，然后使用这些特征来计算和保存模型的预测结果、方差、模型参数、模型状态以及嵌入表示。这表明 `get_features` 函数提供的特征列表是用于构建和评估推荐系统模型的关键输入。

**注意**:
- 虽然 `is_userinfo` 参数在 `get_features` 函数定义中存在，但在当前实现中未直接使用。开发者在使用或修改此函数时，应考虑是否需要根据此参数调整返回的特征列表。
- 特征名称的设计应与实际数据集中的特征对应，开发者可能需要根据实际情况调整特征名称。

**输出示例**:
调用 `get_features` 函数可能会返回如下的三个列表：
- 用户特征：`["user_id"]`
- 项目特征：`["item_id", "feat0", "feat1", "feat2", "feat3", "duration_normed"]`
- 奖励特征：`["watch_ratio_normed"]`

这些列表为推荐系统模型的构建和评估提供了基础特征。
***
### FunctionDef get_df(self, name)
**get_df**: 此函数的功能是根据指定的文件名加载数据框架，并返回处理后的用户特征、物品特征以及类别特征列表。

**参数**:
- name: 字符串类型，默认值为"big_matrix_processed.csv"。指定要加载的数据文件名。

**代码描述**:
`get_df`函数首先根据给定的文件名和预设的数据路径（`DATAPATH`）拼接得到完整的文件路径。接着，调用`get_df_data`函数从该路径下的文件中加载数据，同时指定需要加载的列，包括用户ID、物品ID、时间戳、归一化后的观看比例和归一化后的持续时间。

随后，函数调用`load_category`方法加载项目类别数据，包括类别特征列表和包含这些特征的DataFrame。根据输入参数`name`的值，决定是否仅加载小数据集的用户特征和物品特征。如果`name`等于"big_matrix_processed.csv"，则加载全部数据；否则，仅加载小数据集的数据。

接着，函数分别调用`load_user_feat`和`load_item_feat`方法加载用户特征和物品特征。这两个方法都接受一个布尔参数`only_small`，用于指示是否仅加载小数据集的特征数据。

最后，函数将加载的数据框架与类别特征DataFrame进行连接，并根据物品ID合并。这一步骤是为了将物品的类别特征整合到最终的数据框架中。

在某些情况下，如果需要获取特征的主导性信息，可以调用`get_domination`方法。但在当前的`get_df`实现中，这一部分代码被注释掉了，因此默认情况下不会执行这一步骤。

**注意**:
- 在调用此函数之前，需要确保`DATAPATH`变量已正确设置，并且指定的文件存在于该路径下。
- 此函数依赖于`get_df_data`、`load_user_feat`和`load_item_feat`等方法来加载和处理数据，因此在使用之前需要确保这些方法已正确实现且可用。
- 函数的性能和结果可能受到输入文件大小和指定列的影响，建议仅加载需要的列以优化性能。

**输出示例**:
由于此函数的输出依赖于输入文件和选择的列，因此无法提供一个固定的输出示例。但一般情况下，函数会返回四个对象：处理后的数据框架`df_data`，用户特征`df_user`，物品特征`df_item`，以及类别特征列表`list_feat`。这些返回值为后续的数据分析和模型训练提供了基础数据和特征信息。
***
### FunctionDef get_domination(self)
**get_domination**: 此函数的功能是获取项目特征的主导性信息。

**参数**:
此函数没有显式参数，它通过`self`访问类内的其他方法和属性。

**代码描述**:
`get_domination`函数首先调用`get_df`方法，加载处理后的数据框架，其中包括用户特征、物品特征等信息，但只使用了返回值中的`df_data`和`df_item`。接着，函数检查预定义的数据路径`PRODATAPATH`下是否存在名为`feature_domination.pickle`的文件。如果该文件存在，则直接从文件中加载项目特征的主导性信息；如果不存在，则调用`get_sorted_domination_features`方法计算特征的主导性信息。计算过程中，会考虑特征是否采用多热编码、目标列的名称以及筛选阈值等因素。计算完成后，将结果保存到`feature_domination.pickle`文件中，以便下次直接加载使用。

此函数在项目中的作用是为了获取和缓存特定数据集中的项目特征主导性信息，这对于后续的数据分析和模型训练是非常重要的。例如，在推荐系统或用户行为模型中，了解哪些特征在用户的选择中起主导作用，可以帮助模型更准确地预测用户的行为。

**注意**:
- 确保`PRODATAPATH`路径正确设置，并且有足够的权限读写文件。
- 在首次运行或数据更新后，需要重新计算特征的主导性信息，这可能会消耗一定的计算资源。
- 此函数依赖于`get_df`和`get_sorted_domination_features`方法，确保这些方法能够正确执行是使用此函数的前提。

**输出示例**:
```python
{
    'feature1': [(1, 0.5), (0, 0.5)],
    'feature2': [(0, 0.75), (1, 0.25)]
}
```
此示例展示了一个可能的输出，其中`feature1`和`feature2`分别表示两个不同的特征，列表中的元组第一个元素代表特征值，第二个元素代表该特征值在数据集中的占比。例如，对于`feature1`，值为1的占比为50%，值为0的占比也为50%。
***
### FunctionDef get_item_similarity(self)
**get_item_similarity**: 该函数的功能是获取项目之间的相似度。

**参数**: 该函数没有参数。

**代码描述**: `get_item_similarity` 函数首先尝试从预定的路径（`PRODATAPATH`下的`item_similarity.pickle`文件）加载项目相似度数据。如果该文件存在，则直接从文件中读取项目相似度并返回。如果文件不存在，则调用`load_category`函数加载并处理项目类别数据，得到类别特征列表和包含这些特征的DataFrame。随后，使用`get_similarity_mat`函数计算得到项目相似度矩阵，并将计算结果保存到`item_similarity.pickle`文件中，以便将来使用。最后，函数返回计算或加载的项目相似度矩阵。

在项目中，`get_item_similarity` 函数被 `prepare_train_envs_local` 和 `learn_policy` 两个函数调用。这表明项目相似度数据在训练环境准备和策略学习过程中起到了重要作用。具体来说，在准备训练环境时，项目相似度数据被用于构建模拟环境，以便模拟用户对项目的反馈。在学习策略过程中，项目相似度数据被用于评估策略性能，特别是在用户体验评估方面。

**注意**:
- 确保`PRODATAPATH`变量已正确设置，指向包含`item_similarity.pickle`文件的目录。
- 在首次运行该函数时，需要有足够的权限来创建和写入`item_similarity.pickle`文件。
- 该函数依赖于`load_category`和`get_similarity_mat`两个函数，确保这两个函数的实现正确无误。

**输出示例**: 假设有三个项目的特征集合，该函数可能返回如下的相似度矩阵（以numpy数组形式）:
```
[[1.         0.33333333 0.66666667]
 [0.33333333 1.         0.66666667]
 [0.66666667 0.66666667 1.        ]]
```
这个矩阵表示每个项目特征集合与其他项目特征集合的Jaccard相似度。
***
### FunctionDef get_item_popularity(self)
**get_item_popularity**: 此函数的功能是获取每个项目的受欢迎程度。

**参数**: 此函数没有参数。

**代码描述**: `get_item_popularity` 函数首先尝试从预定义的路径 `PRODATAPATH` 下加载名为 `item_popularity.pickle` 的文件，如果该文件存在，则直接加载并返回项目受欢迎程度的数据。如果文件不存在，则函数会从 `DATAPATH` 路径下读取名为 `big_matrix_processed.csv` 的文件，使用 `get_df_data` 函数加载特定列（`user_id`, `item_id`, `timestamp`, `watch_ratio`），并基于这些数据计算每个项目的受欢迎程度。计算方法是首先过滤出 `watch_ratio` 大于等于1的数据，然后按 `item_id` 分组，对每个项目，计算观看该项目的独立用户数占总用户数的比例作为该项目的受欢迎程度。最后，将计算结果保存到 `item_popularity.pickle` 文件中，以便将来使用。

此函数与项目中其他部分的关系体现在它为其他模块提供了项目受欢迎程度的信息。例如，在 `prepare_train_envs_local` 和 `learn_policy` 函数中，通过调用 `get_item_popularity` 函数获取项目受欢迎程度信息，以此来辅助训练环境的准备和策略学习过程中的决策。

**注意**:
- 在调用此函数之前，需要确保 `PRODATAPATH` 和 `DATAPATH` 路径下的文件存在且格式正确。
- 此函数的性能可能受到数据文件大小的影响，大型数据集可能会导致计算和加载时间较长。
- 如果数据集发生变化，需要删除旧的 `item_popularity.pickle` 文件，以确保受欢迎程度信息是基于最新数据计算的。

**输出示例**: 此函数返回一个 pandas Series 对象，索引为项目ID，值为对应的受欢迎程度。例如：

```
item_id
0    0.05
1    0.10
2    0.15
...
```

此输出示例表示项目ID为0的项目受欢迎程度为0.05，项目ID为1的项目受欢迎程度为0.10，依此类推。
***
### FunctionDef load_user_feat(self, only_small)
**load_user_feat**: 此函数的功能是加载用户特征数据。

**参数**:
- `only_small`: 布尔值，指示是否只加载小数据集的用户特征。

**代码描述**:
`load_user_feat`函数首先确定用户特征数据文件的路径，然后使用`pandas`库的`read_csv`方法加载指定列的数据。这些列包括用户ID、用户活跃度、是否为直播主播、是否为视频作者、关注用户数量范围、粉丝数量范围、好友数量范围、注册天数范围以及18个独热编码特征。

对于某些特征列，函数会将值为"UNKNOWN"的项替换为`chr(0)`，然后使用`LabelEncoder`进行标签编码。如果编码后的类别中包含`chr(0)`或-124，则不对这些列的值进行加一操作；否则，对这些列的值加一以进行调整。

对于18个独热编码特征，函数同样会将缺失值替换为-124，并进行标签编码。编码逻辑与前述相同。

在处理完所有特征后，函数将用户ID设置为数据帧的索引。

如果`only_small`参数为`True`，则函数会调用`get_lbe`方法获取用户ID的标签编码器，并使用这个编码器过滤出小数据集中存在的用户特征，最后返回这部分数据。

如果`only_small`参数为`False`，则直接返回处理后的全部用户特征数据。

此函数与项目中其他部分的关系体现在它为`get_df`方法提供了加载和预处理用户特征数据的功能。`get_df`方法会根据需要加载大数据集或小数据集的用户特征，以便进一步处理和分析。

**注意**:
- 在使用此函数之前，需要确保`DATAPATH`路径下存在`user_features.csv`文件，并且文件格式正确。
- 此函数依赖于`LabelEncoder`进行特征编码，因此需要导入`sklearn.preprocessing`中的`LabelEncoder`。

**输出示例**:
假设`user_features.csv`中有如下数据：

```
user_id,user_active_degree,is_live_streamer,is_video_author,follow_user_num_range,fans_user_num_range,friend_user_num_range,register_days_range,onehot_feat0,...,onehot_feat17
U1,high,Yes,No,1-10,100-500,10-50,30-60,1,...,0
U2,medium,No,Yes,10-50,500-1000,50-100,60-90,0,...,1
```

调用`load_user_feat(only_small=False)`后，返回的数据帧可能如下：

```
          user_active_degree  is_live_streamer  is_video_author  follow_user_num_range  fans_user_num_range  friend_user_num_range  register_days_range  onehot_feat0  ...  onehot_feat17
user_id
U1                         2                 1                0                      1                    2                      1                    1             1  ...              0
U2                         1                 0                1                      2                    3                      2                    2             0  ...              1
```

请注意，实际输出会根据数据和编码过程中的具体值有所不同。
***
### FunctionDef load_item_feat(self, only_small)
**load_item_feat**: 此函数的功能是加载物品特征数据，并根据参数选择是否仅加载小数据集中的物品特征。

**参数**:
- only_small: 布尔类型参数，默认值为False。当设置为True时，函数仅返回小数据集中的物品特征；当设置为False时，返回所有物品特征。

**代码描述**:
`load_item_feat`函数首先调用`load_category`函数来加载项目类别数据，包括类别特征列表和包含这些特征的DataFrame。接着，通过调用`load_video_duration`函数获取视频的平均时长信息，并将这些信息加入到物品特征DataFrame中。如果`only_small`参数为True，则函数会调用`get_lbe`方法获取用户ID和物品ID的标签编码器，然后使用物品ID的编码器筛选出小数据集中的物品特征，并返回这部分数据；如果`only_small`为False，则直接返回所有物品特征。

此函数与项目中其他部分的关系体现在它为`get_df`方法提供了加载物品特征的功能。`get_df`方法在加载用户-物品交互数据时，会根据需要调用`load_item_feat`来获取相应的物品特征数据，以便进行后续的数据处理和分析。

**注意**:
- 在调用此函数之前，需要确保`DATAPATH`路径下存在`item_categories.csv`和`video_duration_normed.csv`文件，这些文件分别用于加载项目类别信息和视频的平均时长信息。
- 如果选择加载小数据集中的物品特征（`only_small=True`），则需要确保`DATAPATH`路径下存在`small_matrix_processed.csv`文件，以及通过`get_lbe`方法生成的用户ID和物品ID的标签编码器文件。

**输出示例**:
假设存在两个物品，其ID分别为1和2，类别特征分别为`[2, 3, 4, 5]`和`[3, 4, 5, 6]`，平均时长分别为120秒和150秒。如果`only_small=False`，函数可能返回如下DataFrame：

```
         feat0  feat1  feat2  feat3  tags          duration_normed
item_id                                                            
1            2      3      4      5  [1, 2, 3, 4]  120
2            3      4      5      6  [2, 3, 4, 5]  150
```

如果`only_small=True`且物品ID为1的物品属于小数据集，则返回的DataFrame可能仅包含该物品的特征信息。
***
### FunctionDef get_lbe(self)
**get_lbe**: 此函数的功能是生成并返回用户ID和物品ID的标签编码器(LabelEncoder)。

**参数**: 此函数不接受任何外部参数，它依赖于类内部的状态和数据路径。

**代码描述**: `get_lbe`函数首先检查`DATAPATH`路径下是否存在`user_id_small.csv`和`item_id_small.csv`文件。如果这些文件不存在，函数会从`small_matrix_processed.csv`文件中读取用户ID(`user_id`)和物品ID(`item_id`)，并提取它们的唯一值生成新的DataFrame对象，分别保存为`user_id_small.csv`和`item_id_small.csv`。如果这些文件已存在，则直接从这些文件中读取数据。接下来，函数使用`LabelEncoder`对用户ID和物品ID进行标签编码，以便将这些非数字的标识符转换为模型训练过程中可以处理的数值形式。最后，函数返回两个标签编码器：`lbe_user`和`lbe_item`，分别用于用户ID和物品ID的编码。

此函数与项目中其他部分的关系体现在它为`load_user_feat`和`load_item_feat`方法提供了必要的用户ID和物品ID的编码器。这些编码器在处理用户特征和物品特征时非常重要，因为它们允许这些方法将文本或分类标识符转换为模型可以理解的数值形式。此外，通过检查文件的存在与否来决定是否需要重新生成编码器，这有助于提高数据处理的效率。

**注意**: 在使用此函数之前，需要确保`DATAPATH`路径下的`small_matrix_processed.csv`文件存在且格式正确。此外，生成的`user_id_small.csv`和`item_id_small.csv`文件将被保存在相同的路径下，这意味着后续的调用将不会重新生成这些文件，除非它们被删除或需要更新。

**输出示例**: 此函数返回两个`LabelEncoder`对象，分别对应于用户ID和物品ID的编码器。例如，如果用户ID有三个唯一值[‘U1’, ‘U2’, ‘U3’]，物品ID有两个唯一值[‘I1’, ‘I2’]，则对应的标签编码器可能将用户ID映射为[0, 1, 2]，将物品ID映射为[0, 1]。
***
### FunctionDef load_mat
**load_mat**: 此函数的功能是加载处理后的小型矩阵数据，并对用户ID和物品ID进行编码，最后返回一个稀疏矩阵及两个标签编码器。

**参数**: 此函数不接受任何外部参数。

**代码描述**: `load_mat` 函数首先构建小型矩阵数据的文件路径，并使用`get_df_data`函数加载指定列（'user_id', 'item_id', 'watch_ratio'）的数据。接着，对超过5的观看比例值进行截断处理，将其设为5。此步骤旨在避免极端值对模型的潜在负面影响。

函数接下来创建两个`LabelEncoder`实例，分别对用户ID和物品ID进行编码。这一步骤是为了将这些分类标签转换为模型可处理的数值形式。

随后，使用`csr_matrix`创建一个压缩稀疏行（CSR）矩阵，其中包含了用户对物品的观看比例，且矩阵的行和列分别对应于编码后的用户ID和物品ID。此矩阵转换为数组形式，并对其中的NaN和无穷值使用观看比例的均值进行填充，以处理可能的数据缺失或异常值。

最后，函数返回处理后的稀疏矩阵、用户ID的标签编码器和物品ID的标签编码器。

在项目中，`load_mat` 函数被`KuaiEnv.py`中的`load_env_data`方法调用，用于加载环境数据，包括用户-物品交互矩阵和相关的标签编码器。这表明`load_mat`函数在项目中扮演着数据预处理和加载的关键角色，为后续的模型训练和推荐系统的构建提供基础数据。

**注意**: 在使用此函数之前，需要确保`DATAPATH`变量已正确设置，且指向包含处理后的小型矩阵数据文件的目录。此外，还需保证`get_df_data`函数能够正常工作，以便加载和预处理数据。

**输出示例**: 函数返回的输出示例可能如下：
- `mat`: 一个二维数组，表示用户-物品的观看比例矩阵。
- `lbe_user`: 用户ID的标签编码器实例。
- `lbe_item`: 物品ID的标签编码器实例。

具体的输出将取决于输入数据的实际情况，包括用户和物品的数量以及它们的观看比例分布。
***
### FunctionDef load_category(tag_label)
**load_category**: 该函数的功能是加载并处理项目类别数据。

**参数**:
- tag_label: 一个字符串参数，默认值为"tags"。用于指定返回的DataFrame中标签列的名称。

**代码描述**:
`load_category`函数主要用于从项目数据路径中加载项目类别信息，处理后返回两个对象：一个是类别特征列表，另一个是包含这些特征的DataFrame。首先，函数通过指定的文件路径（`item_categories.csv`）读取类别数据，然后将特征列（`feat`）中的字符串解析为Python对象。接下来，将这些特征转换为列表，并基于这个列表创建一个DataFrame，其中包含四个特征列（`feat0`, `feat1`, `feat2`, `feat3`），并将索引名称设置为`"item_id"`。DataFrame中的缺失值被替换为-1，并且所有特征值都增加了1以进行调整，最后将特征值转换为整型。此外，函数还会在DataFrame中添加一个由`tag_label`参数指定名称的列，用于存储原始特征列表。

在项目中，`load_category`函数被多个对象调用，包括`get_df`、`get_item_similarity`、`load_item_feat`和`load_env_data`。这些调用点表明，加载和处理的类别特征数据被用于构建推荐系统的不同部分，如获取数据框架、计算项目相似性、加载项目特征和加载环境数据等。这显示了`load_category`函数在整个项目中的核心作用，即提供了一个统一的方法来处理和访问项目类别信息，为后续的数据处理和分析提供基础。

**注意**:
- 确保`DATAPATH`变量已正确设置，指向包含`item_categories.csv`文件的目录。
- `item_categories.csv`文件需要有一个名为`feat`的列，其中包含可解析为Python对象的字符串。

**输出示例**:
```python
(['[1, 2, 3, 4]', '[2, 3, 4, 5]', ...], 
   feat0  feat1  feat2  feat3  tags
item_id                             
0          2      3      4      5  [1, 2, 3, 4]
1          3      4      5      6  [2, 3, 4, 5]
...)
```
此输出示例展示了函数返回的两个对象：一个是解析后的特征列表，另一个是包含这些特征及其处理结果的DataFrame。
***
### FunctionDef load_video_duration
**load_video_duration**: 此函数的功能是加载视频的平均时长信息。

**参数**: 此函数不接受任何参数。

**代码描述**: `load_video_duration` 函数首先尝试从一个名为 "video_duration_normed.csv" 的文件中加载视频的平均时长信息。这个文件应该包含经过归一化处理的视频时长数据。如果该文件存在，则直接使用pandas的read_csv函数读取数据，并从中提取 "duration_normed" 列作为视频的平均时长信息。

如果 "video_duration_normed.csv" 文件不存在，函数将从两个不同的文件 "small_matrix_processed.csv" 和 "big_matrix_processed.csv" 中加载视频时长数据。这两个文件分别包含小数据集和大数据集的视频时长信息。函数使用 `get_df_data` 方法从这两个文件中加载指定的 "item_id" 和 "duration_normed" 列，并将这两个数据集合并。之后，通过对每个 "item_id" 分组并计算其 "duration_normed" 值的平均值，得到每个视频的平均时长信息。最后，将这些信息保存到 "video_duration_normed.csv" 文件中，以便将来使用。

此函数在项目中的作用是提供视频的平均时长信息，这对于视频推荐系统中的特征工程是非常重要的。例如，在 `load_item_feat` 方法中，通过调用 `load_video_duration` 函数获取视频的平均时长信息，并将这些信息与其他特征一起用于构建推荐模型。

**注意**: 
- 确保 "DATAPATH" 路径下的 "video_duration_normed.csv"、"small_matrix_processed.csv" 和 "big_matrix_processed.csv" 文件存在且格式正确。
- 此函数依赖于 `get_df_data` 函数来加载 "small_matrix_processed.csv" 和 "big_matrix_processed.csv" 文件中的数据，因此需要确保 `get_df_data` 函数能够正确执行。
- 函数执行过程中会对数据进行写入操作，因此需要确保有足够的权限来创建和修改 "video_duration_normed.csv" 文件。

**输出示例**: 函数返回一个pandas DataFrame对象，其索引名为 "item_id"，包含一个名为 "duration_normed" 的列，该列记录了每个视频的平均时长信息。
***
### FunctionDef get_similarity_mat(list_feat)
**get_similarity_mat**: 该函数的功能是计算特征列表中各项之间的相似度矩阵。

**参数**:
- list_feat: 特征列表，每个元素代表一个视频的特征集合。

**代码描述**:
`get_similarity_mat` 函数首先尝试从预设的路径加载已存在的相似度矩阵文件。如果该文件存在，则直接从文件中读取相似度矩阵并返回。如果文件不存在，则根据输入的特征列表计算相似度矩阵。计算过程中，首先将特征列表转换为Pandas的DataFrame，然后计算每对特征之间的Jaccard相似度，即两个集合交集的大小除以并集的大小。计算完成后，将相似度矩阵保存到文件中，以便将来使用。

在项目中，`get_similarity_mat` 函数被 `get_item_similarity` 和 `get_saved_distance_mat` 两个函数调用。`get_item_similarity` 函数用于获取物品之间的相似度，如果相似度矩阵已经计算并保存，它会直接加载；否则，它会调用 `get_similarity_mat` 函数计算相似度矩阵。`get_saved_distance_mat` 函数用于获取已保存的距离矩阵，如果需要计算新的距离矩阵，它也会调用 `get_similarity_mat` 函数来获取相似度矩阵，然后根据相似度矩阵计算距离矩阵。

**注意**:
- 在使用该函数之前，确保输入的特征列表格式正确，且每个特征集合中的元素可以进行集合操作。
- 该函数计算的相似度矩阵会被保存到指定路径，以便未来使用，因此需要确保程序有足够的权限访问和修改该路径下的文件。

**输出示例**:
假设有三个视频的特征集合分别为 {1, 2}, {2, 3}, 和 {1, 2, 3}，则该函数可能返回如下的相似度矩阵（以numpy数组形式）:
```
[[1.         0.33333333 0.66666667]
 [0.33333333 1.         0.66666667]
 [0.66666667 0.66666667 1.        ]]
```
这个矩阵表示每个视频特征集合与其他视频特征集合的Jaccard相似度。
***
### FunctionDef get_saved_distance_mat(list_feat, sub_index_list)
**get_saved_distance_mat**: 该函数的功能是获取已保存的视频特征之间的距离矩阵。

**参数**:
- list_feat: 特征列表，每个元素代表一个视频的特征集合。
- sub_index_list: 子索引列表，用于从完整的距离矩阵中提取一个较小的矩阵。

**代码描述**:
`get_saved_distance_mat` 函数首先检查是否提供了子索引列表。如果提供了，函数尝试从预设的路径加载名为 "distance_mat_video_small.csv" 的已保存的距离矩阵文件。如果该文件存在，则直接从文件中读取距离矩阵并返回。在读取文件之前，会输出 "loading small distance matrix..." 以提示正在加载文件，加载完成后输出 "loading completed."。

如果距离矩阵文件不存在，则调用 `get_similarity_mat` 函数计算特征列表中各项之间的相似度矩阵。根据相似度矩阵，通过取其倒数计算距离矩阵，并将计算得到的较小的距离矩阵保存到 "distance_mat_video_small.csv" 文件中，以便将来使用。

如果没有提供子索引列表，函数将返回 None。

在项目中，`get_saved_distance_mat` 函数被 `load_env_data` 函数调用，用于加载环境数据时获取视频特征之间的距离矩阵。这对于基于视频特征进行推荐或相似度计算的应用场景非常重要。

**注意**:
- 确保提供的特征列表和子索引列表格式正确，且子索引列表中的索引有效。
- 该函数依赖于 `get_similarity_mat` 函数来计算相似度矩阵，因此需要确保 `get_similarity_mat` 函数能够正确执行。
- 在使用该函数之前，确保程序有足够的权限访问和修改指定路径下的文件。

**输出示例**:
假设有一个小的距离矩阵如下所示（以Pandas DataFrame形式）:
```
          0         1         2
0  1.000000  0.500000  0.333333
1  0.500000  1.000000  0.666667
2  0.333333  0.666667  1.000000
```
这个矩阵表示每个视频特征集合与其他视频特征集合之间的距离。
***
