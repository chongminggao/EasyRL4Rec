## FunctionDef create_dir(create_dirs)
**create_dir**: 此函数的功能是创建必要的目录。

**参数**:
- create_dirs: 一个包含目录路径的列表，这些目录是需要被创建的。

**代码描述**:
`create_dir` 函数遍历传入的目录路径列表 `create_dirs`，对于列表中的每一个目录路径，首先检查该路径是否已经存在。如果不存在，则使用 `os.mkdir` 方法创建该目录，并通过日志记录器 `logger` 记录创建目录的操作。如果在尝试创建目录时遇到 `FileExistsError` 异常，即目录已存在但是由于某些原因未能通过 `os.path.exists` 检测到，函数会打印一条消息指出该目录已经存在。

在项目中，`create_dir` 函数被 `prepare_dir_log` 函数调用，这发生在 `examples/policy/policy_utils.py` 和 `examples/usermodel/usermodel_utils.py` 中。在这些调用场景中，`create_dir` 被用于创建模型保存路径及其相关的子目录，如日志、参数、模型等存储目录。这些目录的创建是模型训练和日志记录的前提条件，确保了模型输出和日志能够被正确地保存和组织。

**注意**:
- 在使用 `create_dir` 函数时，需要确保传入的 `create_dirs` 参数是一个有效的目录路径列表，且调用该函数的环境中已经正确配置了日志记录器 `logger`。
- 由于 `os.mkdir` 只能创建单层目录，如果需要创建多层嵌套目录，可能需要使用 `os.makedirs` 方法，并适当处理异常。
- 函数中使用了 `%` 格式化字符串来记录日志，这是一种较为传统的格式化方法。在新的Python版本中，可以考虑使用 `.format()` 方法或者 f-string 来提高代码的可读性和效率。
## FunctionDef compute_action_distance(action, actions_hist, env_name, realenv)
**compute_action_distance**: 该函数用于计算当前动作与历史动作之间的距离。

**参数**:
- action: 当前动作，类型为np.ndarray。
- actions_hist: 历史动作集合，类型为np.ndarray。
- env_name: 环境名称，默认为"VirtualTB-v0"。
- realenv: 实际环境对象，默认为None。

**代码描述**:
`compute_action_distance`函数根据不同的环境名称(`env_name`)，采用不同的方法计算当前动作与历史动作之间的距离。该函数支持三种环境："VirtualTB-v0"、"KuaiEnv-v0"和其他环境（默认为"Coat"）。

- 在"VirtualTB-v0"环境下，函数通过计算当前动作与历史动作的差值，然后使用欧几里得距离（Euclidean distance）计算这些差值的范数（norm），从而得到动作之间的距离。
- 在"KuaiEnv-v0"环境下，函数利用实际环境对象`realenv`中的`df_dist_small`属性，根据当前动作和历史动作的索引，从`df_dist_small`中检索出对应的距离。
- 对于其他环境（默认为"Coat"），函数直接从`realenv`的`mat_distance`属性中，根据动作的索引检索出动作之间的距离。

在项目中，`compute_action_distance`函数被`_compute_exposure_effect`方法调用，用于计算给定时间点`t`和动作`action`的曝光效应。该方法首先计算当前动作与历史动作之间的距离，然后基于这个距离和时间差，计算曝光效应，最终根据用户和物品的特定参数调整曝光效应的值。

**注意**:
- 确保传入的`action`和`actions_hist`参数为NumPy数组类型。
- 当使用"KuaiEnv-v0"或其他特定环境时，必须提供`realenv`参数，且该参数应包含计算距离所需的所有必要信息。
- 函数的返回值类型为NumPy数组，表示当前动作与历史动作之间的距离。

**输出示例**:
假设在"VirtualTB-v0"环境下，当前动作为`np.array([1, 2])`，历史动作为`np.array([[1, 1], [2, 2]])`，则函数可能返回`np.array([1., 0.])`，表示当前动作与第一个历史动作的距离为1，与第二个历史动作的距离为0。
## FunctionDef compute_exposure(t_diff, dist, tau)
**compute_exposure**: 该函数用于计算曝光度。

**参数**:
- t_diff: 一个numpy数组，表示时间差。
- dist: 一个numpy数组，表示距离。
- tau: 一个数值，表示时间衰减因子。

**代码描述**:
`compute_exposure`函数接受三个参数：时间差（t_diff）、距离（dist）和时间衰减因子（tau）。该函数首先检查tau是否小于等于0，如果是，则直接返回0，表示没有曝光度。如果tau大于0，函数将计算一个基于时间差、距离和时间衰减因子的指数衰减和。这个和代表了在给定的时间差和距离下，曝光度的量化值。

在项目中，`compute_exposure`函数被`_compute_exposure_effect`方法调用，该方法位于`PenaltyEntExpSimulatedEnv`类中。在这个上下文中，`compute_exposure`用于计算一个动作与历史动作之间的曝光效应。这个效应随后被用来计算环境中的一个惩罚项，该惩罚项与用户和项目的特定参数相乘，并最终影响模拟环境的状态。

**注意**:
- 确保传入的`tau`参数大于0，否则函数将返回0。
- 传入的`t_diff`和`dist`数组应具有相同的长度，因为它们在计算中是一一对应的。

**输出示例**:
假设`t_diff = np.array([1, 2, 3])`，`dist = np.array([0.5, 1.0, 1.5])`，`tau = 2`，那么`compute_exposure`可能返回一个值如`2.315`，这个值表示给定时间差、距离和衰减因子下的曝光度量值。
## FunctionDef softplus_np(x)
**softplus_np函数的功能**: 实现了Softplus激活函数，这是一种用于深度学习中的平滑非线性激活函数。

**参数**:
- x: 输入值，可以是一个数值或者NumPy数组。

**代码描述**:
softplus_np函数接受一个输入x，并使用NumPy库计算Softplus激活函数的值。Softplus激活函数的数学表达式为：log(1 + exp(x))。然而，为了提高数值稳定性，这里采用了一种改进的计算方式：首先计算`np.log1p(np.exp(-np.abs(x)))`，这一步是为了处理x的绝对值，避免在x非常大或非常小的情况下exp(x)造成的数值溢出问题。然后，通过`np.maximum(x, 0)`选取x和0中的最大值，与前一步的结果相加，得到最终的Softplus函数值。

**注意**:
- 输入x可以是单个数值也可以是NumPy数组，这使得该函数可以很方便地应用于标量计算和向量化计算。
- 该函数依赖于NumPy库，因此在使用前需要确保已经正确安装了NumPy。

**输出示例**:
假设输入x为一个NumPy数组`np.array([-3, -1, 0, 1, 3])`，则函数的输出将是一个数组，大致值为`[0.04858735, 0.31326169, 0.69314718, 1.31326169, 3.04858735]`。这个输出展示了Softplus函数平滑处理负值输入的特性，同时保持正值输入的线性增长特性。
## FunctionDef clip0(x)
**clip0**: 此函数的功能是返回输入数组在第0轴上的最大值。

**参数**:
- x: 输入的数组。

**代码描述**:
clip0函数接受一个数组x作为输入，使用numpy库的amax函数计算并返回这个数组在第0轴（通常指的是列方向）上的最大值。这意味着如果输入是多维数组，它将对每一列计算最大值，并返回这些最大值组成的数组。

在项目中，clip0函数被`_compute_pred_reward`方法调用，该方法属于`PenaltyEntExpSimulatedEnv`类，位于`src/core/envs/Simulated_Env/penalty_ent_exp.py`文件中。`_compute_pred_reward`方法计算并返回一个经过惩罚和奖励调整后的最终奖励值。在这个过程中，clip0函数用于确保计算出的奖励值不会因为某些操作导致异常高的值，通过将其“剪切”到一个合理的范围内，以此来维持模型的稳定性和性能。

**注意**:
- clip0函数的使用依赖于numpy库，因此在使用前需要确保numpy库已被正确安装和导入。
- 输入数组x的维度和数据类型会影响到amax函数的行为和返回值，因此在使用clip0函数时应注意输入数据的格式和结构。

**输出示例**:
假设输入一个二维数组[[1, 2], [3, 4]]给clip0函数，它将返回[3, 4]，因为3和4分别是第一列和第二列的最大值。
## FunctionDef find_negative(user_ids, item_ids, neg_u_list, neg_i_list, mat_train, df_negative, is_rand, num_break)
**find_negative**: 该函数的功能是为每个用户-物品对找到一个负样本。

**参数**:
- `user_ids`: 用户ID列表。
- `item_ids`: 物品ID列表。
- `neg_u_list`: 负样本用户ID列表。
- `neg_i_list`: 负样本物品ID列表。
- `mat_train`: 用户-物品交互矩阵。
- `df_negative`: 存储负样本的数据帧。
- `is_rand`: 是否采用随机策略进行负样本采样。
- `num_break`: 尝试次数上限。

**代码描述**:
`find_negative`函数通过遍历用户ID列表和物品ID列表，为每个用户-物品对寻找一个合适的负样本。该函数支持两种模式：随机模式和非随机模式。

- 在随机模式(`is_rand=True`)下，函数通过遍历`neg_u_list`和`neg_i_list`来寻找负样本。对于每个用户-物品对，函数尝试找到一个与原有用户-物品对的交互值小的负样本。如果在`num_break`次尝试后仍未找到满足条件的负样本，将结束循环并记录当前的负样本信息。
- 在非随机模式下，函数通过逐一检查用户的其他物品交互，寻找一个交互值小于原用户-物品对的交互值的负样本。如果在物品ID递增的方向上未找到满足条件的负样本，则在物品ID递减的方向上继续寻找。

无论哪种模式，最终找到的负样本信息（用户ID、物品ID、交互值）都会被记录在`df_negative`中。

在项目中，`find_negative`函数被`negative_sampling`函数调用，用于生成负样本数据。`negative_sampling`函数根据是否需要从已知样本中采样负样本、是否采用随机策略、负样本数量等条件，调用`find_negative`函数生成所需的负样本数据帧，进而用于模型训练或评估。

**注意**:
- 确保`user_ids`和`item_ids`长度相同，它们代表了需要生成负样本的用户-物品对。
- `mat_train`应为一个矩阵或类似矩阵的结构，其中存储了用户和物品的交互信息。
- `df_negative`的初始状态应为一个空的数据帧或具有适当结构（列名包括"user_id", "item_id", 以及交互值）的数据帧，函数将在此基础上填充负样本信息。
- 在使用随机模式时，`neg_u_list`和`neg_i_list`的长度应足够大，以保证有足够的负样本可供选择。
## FunctionDef align_ab(df_a, df_b)
**align_ab**: 该函数的功能是对两个DataFrame进行对齐，使得第一个DataFrame的长度通过重复和随机抽样扩展至与第二个DataFrame的长度相同。

**参数**:
- df_a: 第一个DataFrame，其长度将被扩展。
- df_b: 第二个DataFrame，其长度用于确定df_a需要扩展到的目标长度。

**代码描述**:
此函数首先重置df_a和df_b的索引，确保从0开始且连续，以便后续操作。然后，计算df_b长度与df_a长度的整除结果，即df_a需要重复的次数。通过将df_a自身重复指定次数并合并，形成一个新的DataFrame df_ak，其长度为原始df_a长度的整数倍，但可能仍未达到df_b的长度。为了填补剩余的长度差距，函数随机从df_a中抽取一定数量的行（这个数量等于df_b的长度减去df_ak的长度），形成一个新的DataFrame df_added。最后，将df_ak和df_added合并，形成最终的df_a_res，其长度与df_b相同。函数返回对齐后的df_a_res和原始的df_b。

在项目中，`align_ab`函数被`align_pos_neg`函数调用，用于在正负样本对齐的场景中，根据正负样本集的大小关系，决定如何扩展较小的样本集以匹配较大样本集的大小。这在处理不平衡数据集时特别有用，可以确保模型训练时正负样本的数量一致。

**注意**:
- 确保在调用此函数之前，df_b的长度严格大于df_a的长度，否则函数的行为可能不符合预期。
- 由于函数内部使用了随机抽样，每次调用的结果可能会有所不同。

**输出示例**:
假设有两个DataFrame，df_a和df_b，其中df_a的长度为3，df_b的长度为10。调用`align_ab(df_a, df_b)`后，可能得到的df_a_res的长度将扩展为10，与df_b的长度相同，而df_b保持不变。
## FunctionDef align_pos_neg(df_positive, df_negative, can_divide)
**align_pos_neg**: 该函数的目的是对正负样本集进行对齐，使得它们的数量可以根据指定的条件相匹配。

**参数**:
- df_positive: 正样本的DataFrame。
- df_negative: 负样本的DataFrame。
- can_divide: 布尔值，指示是否可以通过重复正样本集来匹配负样本集的数量。

**代码描述**:
`align_pos_neg`函数首先检查`can_divide`参数。如果为True，函数计算负样本集与正样本集的数量比（neg_K），并断言这个比例是一个整数，以确保正样本可以通过整数倍重复来匹配负样本的数量。然后，将正样本集重复指定倍数并与负样本集对齐。如果`can_divide`为False，函数将根据正负样本集的数量关系，调用`align_ab`函数来对齐样本集。如果负样本集的数量大于正样本集，将正样本集作为第一个参数传递给`align_ab`；反之，则将负样本集作为第一个参数。最终，函数返回对齐后的正负样本集。

在项目中，`align_pos_neg`函数被`negative_sampling`函数调用，用于在负采样过程中对正负样本集进行对齐。这在处理推荐系统或其他机器学习模型时特别有用，可以确保模型训练时正负样本的数量一致，从而提高模型的性能。

**注意**:
- 在调用`align_pos_neg`函数之前，确保传递的正负样本集是正确的，并且`can_divide`参数正确反映了您希望如何处理样本集的对齐。
- `align_pos_neg`函数依赖于`align_ab`函数来处理不能直接通过重复来对齐的样本集，因此确保理解`align_ab`函数的工作原理和限制。

**输出示例**:
假设有两个DataFrame，`df_positive`和`df_negative`，其中`df_positive`的长度为5，`df_negative`的长度为10。调用`align_pos_neg(df_positive, df_negative, can_divide=True)`后，可能得到的`df_pos`的长度将扩展为10，与`df_neg`的长度相同。
## FunctionDef negative_sampling(df_train, df_item, df_user, y_name, is_rand, neg_in_train, neg_K, num_break, sample_neg_popularity)
**negative_sampling**: 该函数的功能是进行负采样。

**参数**:
- `df_train`: 训练数据集的DataFrame。
- `df_item`: 物品信息的DataFrame。
- `df_user`: 用户信息的DataFrame。
- `y_name`: 目标变量的名称。
- `is_rand`: 是否采用随机策略进行负样本采样。
- `neg_in_train`: 是否仅从已知样本中采负样本。
- `neg_K`: 负样本与正样本的比例。
- `num_break`: 尝试找到负样本的最大次数。
- `sample_neg_popularity`: 是否根据物品的流行度进行负样本采样。

**代码描述**:
`negative_sampling`函数首先判断是否仅从已知样本中采负样本。如果是，则从`df_train`中分离出正负样本，并根据`neg_K`参数复制负样本，以达到指定的正负样本比例。接着，通过`align_pos_neg`函数对正负样本进行对齐。

如果不仅从已知样本中采负样本，函数将构建用户-物品交互矩阵，并根据`neg_K`参数循环生成负样本。在每次循环中，根据是否考虑物品流行度，选择不同的方式生成负样本用户ID和物品ID列表。然后，调用`find_negative`函数为每个用户-物品对找到一个负样本。最后，通过`align_pos_neg`函数对正负样本进行对齐。

在项目中，`negative_sampling`函数被`load_dataset_train`和`load_dataset_train_IPS`函数调用，用于生成负样本数据，进而用于模型训练或评估。

**注意**:
- 确保`df_train`、`df_item`和`df_user`具有正确的结构和数据。
- `y_name`参数应正确指向目标变量的列名。
- 在调用`negative_sampling`函数之前，应理解其参数的含义和作用，以便根据实际需求进行配置。

**输出示例**:
调用`negative_sampling`函数可能会返回两个DataFrame：`df_pos`和`df_neg`，分别代表对齐后的正样本和负样本集。例如，如果`neg_K=5`，那么对于每个正样本，将有5个负样本与之对应。
## FunctionDef compute_input_dim(feature_columns, include_sparse, include_dense, feature_group)
**compute_input_dim**: 此函数的功能是计算输入特征的维度总和。

**参数**:
- `feature_columns`: 特征列列表，包含稀疏和密集特征列。
- `include_sparse`: 布尔值，指示是否包含稀疏特征的维度，默认为True。
- `include_dense`: 布尔值，指示是否包含密集特征的维度，默认为True。
- `feature_group`: 布尔值，指示是否将稀疏特征视为一个组，默认为False。

**代码描述**:
`compute_input_dim`函数通过分析传入的`feature_columns`参数，计算出模型输入所需的总维度。首先，它将特征列分为稀疏特征列和密集特征列两类。对于密集特征列，直接计算其维度之和。对于稀疏特征列，根据`feature_group`参数的值，要么计算所有稀疏特征的嵌入维度之和，要么仅计算稀疏特征列的数量。最后，根据`include_sparse`和`include_dense`参数决定是否将稀疏和密集特征的维度加入到最终的维度总和中。

在项目中，此函数被多个用户模型对象调用，例如`UserModel_MMOE`、`UserModel_Pairwise`和`UserModel_Pairwise_Variance`等，用于初始化模型时计算输入层的维度。这些用户模型对象通过传入不同的特征列配置，使用`compute_input_dim`函数来确定其DNN网络或其他组件的输入大小。

**注意**:
- 确保传入的`feature_columns`参数正确反映了模型预期接收的特征列。
- `include_sparse`和`include_dense`参数应根据模型设计适当设置，以避免不必要的计算开销。

**输出示例**:
假设有3个稀疏特征列，每个的嵌入维度分别为4、6、8，以及2个密集特征列，维度分别为10和20。如果`include_sparse`和`include_dense`都设置为True，且`feature_group`为False，则`compute_input_dim`的返回值将是48（稀疏特征维度之和18 + 密集特征维度之和30）。
