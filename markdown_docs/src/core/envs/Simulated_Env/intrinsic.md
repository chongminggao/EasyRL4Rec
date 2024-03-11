## ClassDef IntrinsicSimulatedEnv
**IntrinsicSimulatedEnv**: IntrinsicSimulatedEnv类的功能是在模拟环境中加入内在奖励机制，以增强模型的探索性和多样性。

**属性**:
- `ensemble_models`: 用于评估的模型集合。
- `env_task_class`: 环境任务的类。
- `task_env_param`: 任务环境参数，为字典格式。
- `task_name`: 任务名称。
- `predicted_mat`: 预测矩阵，用于预测奖励。
- `item_similarity`: 物品相似度矩阵。
- `item_popularity`: 物品流行度。
- `lambda_diversity`: 多样性权重。
- `lambda_novelty`: 新颖性权重。

**代码描述**:
IntrinsicSimulatedEnv类继承自BaseSimulatedEnv类，通过添加内在奖励机制（包括多样性和新颖性奖励）来增强模型的探索性和多样性。在初始化方法中，除了从BaseSimulatedEnv继承的参数外，还接受物品相似度矩阵、物品流行度、多样性权重和新颖性权重作为输入，用于计算内在奖励。`_compute_pred_reward`方法用于计算预测奖励，并结合内在奖励计算最终奖励。`_cal_diversity`和`_cal_novelty`方法分别用于计算动作的多样性奖励和新颖性奖励。

在项目中，IntrinsicSimulatedEnv类通过`prepare_train_envs_local`函数被实例化，并用于训练环境的构建。该类利用物品相似度和流行度信息，通过调整多样性和新颖性权重来控制内在奖励的影响，从而促进模型在探索过程中考虑到物品的多样性和新颖性，提高推荐系统的整体性能。

**注意**:
- 使用IntrinsicSimulatedEnv类时，需要确保传入的物品相似度矩阵和物品流行度数据准确无误。
- 多样性权重和新颖性权重的设置应根据具体任务和数据集进行调整，以达到最佳效果。

**输出示例**:
调用`_compute_pred_reward`方法可能返回的示例输出为：
```python
final_reward = 5.2
```
其中`final_reward`表示考虑内在奖励后的最终奖励值。
### FunctionDef __init__(self, ensemble_models, env_task_class, task_env_param, task_name, predicted_mat, item_similarity, item_popularity, lambda_diversity, lambda_novelty)
**__init__**: 此函数用于初始化IntrinsicSimulatedEnv类的实例。

**参数**:
- **ensemble_models**: 用于模拟环境的模型集合。
- **env_task_class**: 环境任务的类。
- **task_env_param**: 任务环境参数，以字典形式提供。
- **task_name**: 任务的名称，以字符串形式提供。
- **predicted_mat**: 预测矩阵，默认为None。
- **item_similarity**: 物品相似度，默认为None。
- **item_popularity**: 物品流行度，默认为None。
- **lambda_diversity**: 多样性的权重系数，默认为0.1。
- **lambda_novelty**: 新颖性的权重系数，默认为0.1。

**代码描述**:
此函数是IntrinsicSimulatedEnv类的构造函数，负责初始化类的实例。它首先调用父类的构造函数来初始化继承自父类的属性，包括模型集合、环境任务类、任务环境参数和任务名称以及预测矩阵。接着，它设置了几个与推荐系统内在特性相关的属性，包括物品相似度(`item_similarity`)、物品流行度(`item_popularity`)、多样性权重(`lambda_diversity`)和新颖性权重(`lambda_novelty`)。这些属性用于后续在模拟环境中评估推荐系统的性能时，考虑推荐列表的多样性和新颖性。

**注意**:
- `predicted_mat`参数是可选的，如果在实例化时未提供，将默认为None。这个参数通常用于提供预测的用户-物品评分矩阵。
- `item_similarity`和`item_popularity`也是可选参数，它们分别用于描述物品之间的相似度和物品的流行度，这对于评估推荐列表的多样性和新颖性非常重要。
- `lambda_diversity`和`lambda_novelty`是权重系数，用于调整多样性和新颖性在推荐系统性能评估中的重要性。这两个参数的默认值是0.1，但可以根据具体需求进行调整。
***
### FunctionDef _compute_pred_reward(self, action)
**_compute_pred_reward**: 该函数用于计算给定动作的预测奖励，并结合内在奖励（多样性和新颖性）以及最小奖励值来计算最终奖励。

**参数**:
- action: 执行的动作，其类型和具体含义根据环境的不同而有所差异。

**代码描述**:
_compute_pred_reward 函数首先根据环境名称（env_name）判断当前环境，并据此采取不同的策略来计算预测奖励（pred_reward）。如果环境为 "VirtualTB-v0"，则会根据当前用户状态、奖励、总轮数和动作等信息构造特征，通过用户模型预测出奖励值。预测奖励值会被限制在0到10之间。对于其他环境（例如 "KuaiEnv-v0"），预测奖励可能直接从某个预先计算的矩阵中获取。

接下来，函数调用 `_cal_diversity` 和 `_cal_novelty` 函数分别计算给定动作的多样性奖励（div_reward）和新颖性奖励（nov_reward）。这两个函数分别评估动作的多样性和新颖性，并返回相应的得分。内在奖励是通过将多样性奖励和新颖性奖励与各自的权重（lambda_diversity 和 lambda_novelty）相乘并相加得到的。

最终奖励（final_reward）是通过将预测奖励、内在奖励和最小奖励值（MIN_R）相加得到的。这样的设计旨在综合考虑预测奖励和内在奖励（多样性和新颖性），以促进探索和利用之间的平衡。

**注意**:
- 预测奖励的计算可能依赖于特定的用户模型和环境设置，需要根据实际情况进行适配。
- 多样性和新颖性奖励的权重（lambda_diversity 和 lambda_novelty）是影响最终奖励计算的关键参数，应根据任务目标进行调整。
- 最小奖励值（MIN_R）用于调整奖励的基线，可能需要根据不同的环境和任务进行调整。

**输出示例**:
假设预测奖励为5，多样性奖励为0.3，新颖性奖励为0.5，lambda_diversity为0.2，lambda_novelty为0.3，MIN_R为1，则最终奖励的计算过程如下：
intrinsic_reward = 0.2 * 0.3 + 0.3 * 0.5 = 0.21
final_reward = 5 + 0.21 - 1 = 4.21
因此，该函数返回的最终奖励为4.21。
***
### FunctionDef _cal_diversity(self, action)
**_cal_diversity**: 该函数用于计算动作的多样性得分。

**参数**:
- action: 执行的动作，其类型和具体含义根据环境的不同而有所差异。

**代码描述**:
_cal_diversity 函数旨在评估给定动作与历史动作集合的多样性。首先，如果环境任务（env_task）具有属性 `lbe_item`，则使用该属性的 `inverse_transform` 方法将动作转换为其原始表示（即 `p_id`）。如果没有 `lbe_item` 属性，动作本身即作为其原始表示。接下来，函数计算当前动作与历史动作之间的多样性得分。如果历史动作的数量小于或等于1，多样性得分直接返回0.0。否则，对于历史动作中的每一个动作，如果存在 `lbe_item` 属性，同样使用 `inverse_transform` 方法获取其原始表示（即 `q_id`），然后计算当前动作与该历史动作的相似度差（1 - 相似度），并累加。最后，将累加的总和除以历史动作数量减1，得到平均多样性得分，并返回该得分。

在项目中，_cal_diversity 函数被 `_compute_pred_reward` 函数调用，用于计算给定动作的内在奖励中的多样性部分。这是通过将动作的多样性得分与预设的多样性权重（lambda_diversity）相乘，然后与其他内在奖励（如新颖性奖励）相加，最终与预测奖励相加，从而影响最终的奖励值。

**注意**:
- 该函数假设 `item_similarity` 是一个可访问的属性，其中包含了不同动作之间相似度的信息。
- 动作的原始表示（p_id 或 q_id）的获取方式可能因环境的不同而有所变化，这需要在实际使用时根据环境特性进行适配。

**输出示例**:
假设历史动作集合中有3个动作，与当前动作的相似度分别为0.9, 0.8, 0.7，则多样性得分计算如下：
div = (1 - 0.9) + (1 - 0.8) + (1 - 0.7) = 0.1 + 0.2 + 0.3 = 0.6
div /= 3 - 1 = 0.6 / 2 = 0.3
因此，该函数返回的多样性得分为0.3。
***
### FunctionDef _cal_novelty(self, action)
**_cal_novelty**: 该函数的功能是计算给定动作的新颖性分数。

**参数**:
- action: 执行的动作，可以是一个整数或其他形式，具体取决于环境任务的配置。

**代码描述**:
函数`_cal_novelty`旨在计算并返回一个给定动作的新颖性分数。它首先检查环境任务(`self.env_task`)是否具有`lbe_item`属性。如果有，这意味着动作需要通过`lbe_item`的`inverse_transform`方法转换为对应的项目ID(`p_id`)。如果没有`lbe_item`属性，动作本身被视为项目ID。接下来，通过项目ID索引到`self.item_popularity`字典中获取该项目的流行度(`item_pop`)。新颖性分数(`nov`)是通过对项目流行度加上一个非常小的数(1e-10)取负对数得到的。这样做是为了避免对数运算时分母为零的情况，并且通过取负对数，确保了流行度越低的项目，其新颖性分数越高。

在项目中，`_cal_novelty`函数被`_compute_pred_reward`函数调用，用于计算给定动作的内在奖励中的新颖性成分。`_compute_pred_reward`函数结合预测奖励、多样性奖励和新颖性奖励来计算最终的奖励值，其中新颖性奖励是通过调用`_cal_novelty`函数获得的。

**注意**:
- 在使用`_cal_novelty`函数时，需要确保`self.item_popularity`已经被正确初始化，且包含了所有可能动作的流行度信息。
- 该函数假设较低的项目流行度对应较高的新颖性，这一点在设计奖励机制时需要考虑。

**输出示例**:
假设某个动作对应的项目流行度为0.0001，那么`_cal_novelty`函数的返回值大约为9.2103。这个值是通过计算`-np.log(0.0001 + 1e-10)`得到的。
***
