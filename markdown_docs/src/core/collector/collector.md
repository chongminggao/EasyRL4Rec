## ClassDef Collector
**Collector**: Collector 类的功能是在给定策略、环境和可选的数据缓冲区下，收集交互数据。

**属性**:
- `policy`: 使用的策略实例，通常是一个基于`tianshou.policy.BasePolicy`的策略类实例。
- `env`: 交互环境，可以是`gym.Env`的实例或者是`tianshou.env.BaseVectorEnv`的实例。
- `buffer`: 数据缓冲区，用于存储交互数据，是`tianshou.data.ReplayBuffer`的实例。如果设置为None，则不存储数据。
- `preprocess_fn`: 数据预处理函数，用于在数据被添加到缓冲区之前进行预处理。
- `exploration_noise`: 是否在动作中添加探索噪声。
- `remove_recommended_ids`: 是否移除推荐的ID，默认为False。
- `force_length`: 强制交互长度，默认为0。

**代码描述**:
Collector 类是一个用于在强化学习中收集交互数据的工具。它通过与环境进行交互，根据给定的策略生成动作，收集状态、动作、奖励等信息，并可选地将这些信息存储在数据缓冲区中。此外，Collector 类支持在动作选择过程中添加探索噪声，以及在数据添加到缓冲区之前进行自定义的预处理操作。

Collector 类在项目中被多个场景调用，例如在不同的强化学习算法实现中用于训练和测试阶段的数据收集。通过`CollectorSet`类，可以针对不同的测试环境创建多个Collector实例，以实现更灵活的数据收集策略。

**注意**:
- 使用Collector时，请确保给定的环境具有时间限制，特别是在使用`n_episode`收集选项时。
- 在使用探索噪声时，需要确保策略类支持探索噪声的添加。
- 数据预处理函数`preprocess_fn`的使用可以极大地影响数据的质量和后续学习的效果，因此需要根据具体任务仔细设计。

**输出示例**:
由于Collector主要用于数据收集，其直接输出通常是对环境的观察、动作、奖励等信息的记录，以及这些信息的存储（如果指定了数据缓冲区）。具体的输出格式依赖于环境和策略的实现，以及是否进行了数据预处理。例如，一个可能的输出是一个包含多个交互步骤信息的数据批次，其中每个步骤包含观察值、动作、奖励等字段。
### FunctionDef __init__(self, policy, env, buffer, preprocess_fn, exploration_noise, remove_recommended_ids, force_length)
**__init__**: 此函数用于初始化Collector对象。

**参数**:
- `policy`: BasePolicy类型，定义了与环境交互时所采用的策略。
- `env`: gym.Env或BaseVectorEnv类型，代表与策略交互的环境。
- `buffer`: 可选的ReplayBuffer对象，用于存储环境的交互数据。
- `preprocess_fn`: 可选的Callable对象，用于预处理收集到的数据。
- `exploration_noise`: 布尔值，指示是否在策略执行时添加探索噪声。
- `remove_recommended_ids`: 布尔值，用于控制是否移除推荐的ID。
- `force_length`: 整数，用于强制设定某个长度。

**代码描述**:
`__init__`函数是`Collector`类的构造函数，负责初始化数据收集器的各项参数和状态。首先，它会检查环境类型，如果是单个环境且没有`__len__`属性，则会发出警告并将其包装为`DummyVectorEnv`。接着，根据环境数量初始化`env_num`属性，并设置探索噪声、策略、预处理函数和动作空间等属性。

此函数还调用了`_assign_buffer`私有方法来分配或校验传入的缓冲区对象，确保后续的数据收集过程能够顺利进行。此外，通过调用`reset`方法，确保在收集器实例化时，环境和相关数据处于初始状态，为数据收集做好准备。

**注意**:
- 如果传入的环境是单个环境而非向量环境，将自动被包装为`DummyVectorEnv`，以统一处理逻辑。
- 在初始化过程中，`_assign_buffer`方法用于校验和分配缓冲区，确保其与环境数量兼容，这一点对于后续正确收集和存储数据至关重要。
- `reset`方法在构造函数中被调用，以确保收集器的状态和数据从一个干净的状态开始，这对于避免潜在的数据污染非常重要。
- 参数`remove_recommended_ids`和`force_length`提供了额外的控制选项，但在文档中未详细说明其具体用途，开发者在使用时需要根据实际情况和项目需求进行适当的处理。
***
### FunctionDef _assign_buffer(self, buffer)
**_assign_buffer**: 此函数的功能是校验并分配合适的缓冲区给收集器。

**参数**:
- `buffer`: 可选的`ReplayBuffer`对象，用于存储环境的交互数据。

**代码描述**:
`_assign_buffer`函数是`Collector`类的一个私有方法，用于为环境交互数据收集器分配一个合适的缓冲区。这个函数首先检查传入的`buffer`参数。如果`buffer`为`None`，则会根据当前环境数量(`env_num`)创建一个`VectorReplayBuffer`对象作为缓冲区。如果`buffer`是`ReplayBufferManager`的实例，会进一步检查其`buffer_num`是否不小于环境数量(`env_num`)，并且如果`buffer`是`CachedReplayBuffer`的实例，还会检查其`cached_buffer_num`是否也不小于环境数量。对于其他类型的`buffer`（即`ReplayBuffer`或`PrioritizedReplayBuffer`），会检查其最大容量(`maxsize`)是否大于0，并且如果环境数量大于1，会抛出`TypeError`，提示用户使用对应的向量类型缓冲区（`VectorReplayBuffer`或`PrioritizedVectorReplayBuffer`）。

这个函数的设计考虑到了不同类型的缓冲区和环境数量的兼容性，确保了数据收集过程的有效性和灵活性。通过对缓冲区类型和容量的校验，保证了数据收集器能够正确地处理来自多个环境的交互数据。

在项目中，`_assign_buffer`函数被`Collector`类的构造函数`__init__`调用。在`Collector`对象初始化时，会通过这个函数来分配或校验传入的缓冲区对象，确保后续的数据收集过程能够顺利进行。这体现了代码设计中的模块化和责任分离原则，使得缓冲区的管理和数据收集逻辑更加清晰和可维护。

**注意**:
- 当传入的`buffer`为`None`时，会根据环境数量自动创建一个新的向量类型缓冲区。
- 如果环境数量大于1，而传入的缓冲区类型不支持多环境数据收集，函数会抛出`TypeError`，提示需要使用向量类型的缓冲区。这要求开发者在使用多环境收集数据时，需要注意缓冲区类型的选择。
***
### FunctionDef reset(self, reset_buffer, gym_reset_kwargs)
**reset**: 此函数用于重置环境、统计信息、当前数据以及可能的回放内存。

**参数**:
- `reset_buffer`: 布尔值，默认为True。决定是否重置附加到收集器的回放缓冲区。
- `gym_reset_kwargs`: 字典类型，可选。用于传递给环境的reset函数的额外关键字参数。默认值为None。

**代码描述**:
`reset`函数是`Collector`类的一个重要成员方法，它负责在强化学习训练或评估前，初始化或重置环境、统计信息、当前数据以及回放缓冲区（如果有）。此函数首先创建一个空的`Batch`对象用于初始化`self.data`，确保`self.data`支持切片操作。然后，调用`reset_env`方法重置环境，并根据`reset_buffer`参数的值决定是否调用`reset_buffer`方法来重置回放缓冲区。最后，调用`reset_stat`方法重置统计变量。

在项目中，`reset`方法被`Collector`类的构造函数`__init__`调用，以确保在收集器实例化时，环境和相关数据处于初始状态。此外，`reset`方法还可能在需要重新开始数据收集过程时被显式调用，以确保从干净的状态开始。

`reset`方法通过调用`reset_env`、`reset_buffer`和`reset_stat`三个方法，实现了环境重置、回放缓冲区重置（如果需要）和统计信息重置的功能。这种设计使得`reset`方法在功能上具有较高的灵活性和可扩展性，能够适应不同的数据收集需求。

**注意**:
- 在调用`reset`方法时，应根据实际需求设置`reset_buffer`参数，以决定是否需要重置回放缓冲区。默认情况下，此参数为True，意味着会重置回放缓冲区。
- `gym_reset_kwargs`参数允许用户传递额外的关键字参数给环境的`reset`函数，这在需要对环境重置行为进行特殊配置时非常有用。如果不需要传递额外参数，此参数可以保持默认值None。
- 在使用`reset`方法时，应注意其会重置收集器的所有状态和数据，包括环境状态、统计信息和回放缓冲区（如果有）。因此，在调用此方法前，应确保已经适当处理或保存了需要保留的数据。
***
### FunctionDef reset_stat(self)
**reset_stat**: 此函数的功能是重置统计变量。

**参数**: 此函数没有参数。

**代码描述**: `reset_stat` 函数是`Collector`类的一个成员方法，用于将收集器的统计变量重置为其初始状态。具体来说，它将`collect_step`（收集步骤数）、`collect_episode`（收集的剧集数）和`collect_time`（收集的时间）这三个变量重置为0或0.0。这是在收集过程中跟踪和管理收集器状态的重要步骤，确保每次收集操作开始时，统计数据都是准确的。

在项目中，`reset_stat`函数被`reset`方法调用。`reset`方法负责重置环境、统计数据、当前数据以及可能的回放内存。在`reset`方法中，`reset_stat`被调用来确保统计变量被重置，这是环境和收集器重置流程的一部分。通过这种方式，`reset_stat`函数与`reset`方法一起，确保了收集器在每次重置操作后都能以一致的状态开始新的数据收集周期。

**注意**: 使用`reset_stat`方法时，不需要传递任何参数。此方法的调用应该与收集器的其他重置操作（如环境重置和回放缓冲区重置）结合使用，以确保收集器的状态完全初始化。在进行新的数据收集之前调用`reset_stat`，可以避免统计数据的累积，确保收集到的数据反映了最新的环境状态。
***
### FunctionDef reset_buffer(self, keep_statistics)
**reset_buffer**: 该函数用于重置数据缓冲区。

**参数**:
- keep_statistics: 布尔值，默认为False。决定在重置缓冲区时是否保留统计信息。

**代码描述**:
`reset_buffer`函数是`Collector`类的一个方法，用于重置与数据收集相关的缓冲区。当调用此方法时，它会调用缓冲区对象的`reset`方法，该方法接受一个名为`keep_statistics`的参数。此参数用于指示在重置缓冲区的过程中是否需要保留统计信息。如果`keep_statistics`为True，则在重置缓冲区时会保留统计信息；如果为False，则不保留。

在项目中，`reset_buffer`方法被`reset`方法调用。`reset`方法是`Collector`类中用于重置环境、统计信息、当前数据以及可能的重放内存的方法。在`reset`方法的实现中，根据`reset_buffer`参数的值（默认为True），决定是否调用`reset_buffer`方法来重置缓冲区。这表明`reset_buffer`方法在整个数据收集和环境重置流程中起着重要的作用，特别是在需要清除旧的数据以开始新一轮数据收集时。

**注意**:
- 在使用`reset_buffer`方法时，需要根据实际情况决定是否保留统计信息。例如，在一些场景中，保留统计信息可能对分析和调试有帮助，而在其他场景中，可能希望完全重置所有信息以确保数据的独立性。
- 调用`reset_buffer`之前，应确保所有需要保留的数据已经被适当处理或保存，因为一旦执行重置操作，未保存的数据将无法恢复。
***
### FunctionDef reset_state_tracker(self, global_ids)
**reset_state_tracker**: 该函数用于重置状态跟踪器。

**参数**:
- **global_ids** (可选): 用于指定需要重置状态的环境ID，如果不提供，则默认对所有环境进行操作。

**代码描述**:
`reset_state_tracker` 函数是Collector类的一个成员方法，其主要功能是重置状态跟踪器。在某些情况下，例如环境重置或者开始新的一轮数据收集时，可能需要将之前的状态信息清除，以确保数据的准确性和一致性。该函数通过调用预处理函数（如果存在的话）来实现状态的重置。

当调用此函数时，首先会检查`preprocess_fn`是否被定义。`preprocess_fn`是一个预处理函数，可以在Collector对象初始化时指定。如果`preprocess_fn`存在，那么会调用这个函数，并传入以下参数：
- `dim_batch`：表示环境的数量，通过`self.env_num`获取。
- `reset`：设置为True，表示需要进行重置操作。
- `env_id`：传入的`global_ids`参数，指定需要重置状态的环境ID。如果`global_ids`为None，则表示对所有环境进行重置。

通过这种方式，`reset_state_tracker`函数能够灵活地处理不同的重置需求，无论是针对单个环境还是多个环境的状态重置。

**注意**:
- 在使用`reset_state_tracker`函数时，需要确保`preprocess_fn`已经正确定义并能接受`dim_batch`、`reset`和`env_id`这三个参数。否则，调用此函数可能会导致错误。
- 如果不需要对特定的环境进行状态重置，可以不传递`global_ids`参数，这样会对所有环境进行重置操作。
- 该函数不返回任何值。
***
### FunctionDef reset_env(self, gym_reset_kwargs)
**reset_env**: 此函数的功能是重置所有环境。

**参数**:
- `gym_reset_kwargs`: 可选的字典类型参数，用于传递给环境的reset函数的额外关键字参数。默认值为None。

**代码描述**:
`reset_env`函数主要用于在强化学习中重置环境，以开始新的一轮训练或评估。函数首先检查是否提供了`gym_reset_kwargs`参数，如果没有提供，则使用空字典作为默认值。随后，该函数调用环境的`reset`方法，并将`gym_reset_kwargs`作为参数传递，以便于在重置环境时可以传入特定的配置。环境重置后，会返回观察值(obs)和额外信息(info)，其中观察值代表了环境的初始状态。

在环境重置后，函数将观察值、额外信息、上一次的奖励（初始化为零向量）以及是否为起始状态（初始化为布尔值True的向量）保存到`self.data`中，以便于后续的数据收集和处理。

从项目的调用情况来看，`reset_env`函数被`reset`和`collect`两个函数调用。在`reset`函数中，`reset_env`被用于重置环境并开始新的训练或评估周期，在重置环境之前，`reset`函数会重置当前数据和统计信息，并根据参数决定是否重置回放缓冲区。在`collect`函数中，`reset_env`用于在收集数据过程中，当某些环境达到终止状态时，对这些环境进行重置，以便继续数据收集过程。

**注意**:
- 在使用`reset_env`函数时，需要注意`gym_reset_kwargs`参数的设定，因为不同的环境可能需要不同的重置参数。如果环境重置时需要特定的配置，请确保正确设置此参数。
- `reset_env`函数假设环境的`reset`方法返回的观察值是用户的ID，这一点在使用时需要特别注意，确保环境的返回值与此假设相匹配。
***
### FunctionDef _reset_state(self, id)
**_reset_state**: 该函数用于重置隐藏状态：self.data.state[id]。

**参数**:
- **id**: 可以是一个整数或一个整数列表，指定需要重置状态的ID。

**代码描述**:
`_reset_state` 函数是`Collector`类的一个私有方法，用于重置指定ID的隐藏状态。这个功能在强化学习中尤为重要，因为它能够帮助模型在每次迭代或者环境重置时忘记之前的状态，从而保证模型的泛化能力和稳定性。

具体来说，该函数首先检查`self.data.policy`是否有`hidden_state`属性。如果有，它将获取这个隐藏状态的引用。然后，根据隐藏状态的数据类型（`torch.Tensor`、`np.ndarray`或`Batch`），采取不同的操作来重置状态。对于`torch.Tensor`，它会将指定ID的状态置零；对于`np.ndarray`，如果数据类型为`object`，则将状态设置为`None`，否则置零；对于`Batch`，它会调用`empty_`方法来清空指定ID的状态。

在项目中，`_reset_state`函数被`collect`方法调用。在`collect`方法中，当环境中的某个或某些状态完成（比如达到终止条件或被截断）时，需要对这些状态进行重置，以便于下一轮的数据收集。`_reset_state`通过重置这些完成状态的隐藏状态，为新的数据收集周期做准备，确保数据的准确性和模型的有效学习。

**注意**:
- `_reset_state`是一个内部方法，意味着它仅在`Collector`类内部使用，不应该被类外部直接调用。
- 传递给`_reset_state`的ID应该正确对应于需要重置状态的环境或实体，错误的ID可能会导致状态重置不正确，影响模型的学习效果。
***
### FunctionDef _reset_env_with_ids(self, local_ids, global_ids, gym_reset_kwargs)
**_reset_env_with_ids**: 此函数的功能是根据局部和全局ID重置环境，并更新相关数据。

**参数**:
- local_ids: 局部ID列表或NumPy数组，表示需要重置的环境的局部索引。
- global_ids: 全局ID列表或NumPy数组，表示需要重置的环境的全局索引。
- gym_reset_kwargs: 可选的字典，包含传递给环境reset函数的额外关键字参数。默认为None。

**代码描述**:
此函数首先检查`gym_reset_kwargs`参数是否为None，如果是，则将其设置为空字典。然后，使用`global_ids`和任何额外的关键字参数调用环境的`reset`方法，该方法返回重置后的观察结果和信息。接着，函数使用局部ID更新Collector对象的数据属性，包括观察结果、信息、上一步的奖励以及是否为起始状态的标志。上一步的奖励被设置为零，而起始状态的标志被设置为True，表示这些环境处于新的起始状态。

在项目中，`_reset_env_with_ids`函数被`collect`方法调用，用于在环境达到终止状态时重置环境。在数据收集过程中，当某些环境完成一个回合或达到终止条件时，`collect`方法会通过调用`_reset_env_with_ids`函数来重置这些环境，并准备它们以开始新的回合。这样做是为了确保数据收集可以连续进行，同时保持环境的正确状态。

**注意**:
- 使用此函数时，需要确保`local_ids`和`global_ids`正确对应到Collector对象管理的环境中，以避免索引错误。
- 在调用此函数之前，应当确保任何需要作为`gym_reset_kwargs`传递的额外参数都已正确准备，以确保环境能够根据这些参数正确重置。
***
### FunctionDef collect(self, n_step, n_episode, random, render, no_grad, gym_reset_kwargs, is_train)
**collect**: 此函数用于收集指定数量的步骤或者回合的数据。

**参数**:
- `n_step`: 指定要收集的步骤数。
- `n_episode`: 指定要收集的回合数。
- `random`: 是否使用随机策略来收集数据，默认为False。
- `render`: 在连续渲染帧之间的睡眠时间，默认为None（不渲染）。
- `no_grad`: 是否在policy.forward()中保留梯度，默认为True（不保留梯度）。
- `gym_reset_kwargs`: 传递给环境reset函数的额外关键字参数，默认为None。
- `is_train`: 指示当前收集操作是训练模式还是评估模式，默认为True。

**代码描述**:
`collect`函数的主要目的是根据指定的步骤数(`n_step`)或回合数(`n_episode`)来收集环境数据。函数开始时会检查环境是否为异步环境，并确保只指定了`n_step`或`n_episode`中的一个。随后，根据`n_step`或`n_episode`的设置，初始化环境ID，并在必要时发出警告。函数通过循环执行动作选择、环境交互、数据收集和状态更新，直到达到指定的步骤数或回合数。在此过程中，可以选择是否使用随机策略、是否渲染环境以及是否在策略前向传播中保留梯度。此外，还可以通过`gym_reset_kwargs`参数传递额外的环境重置参数。收集完成后，函数会计算并返回收集过程的统计信息，如回合数、步骤数、奖励数组、回合长度等。

在项目中，`collect`函数与多个对象和方法有关联，例如`reset_env`用于重置所有环境，`_reset_state`用于重置指定ID的隐藏状态，`_reset_env_with_ids`根据局部和全局ID重置环境，并更新相关数据。此外，还与`RecPolicy`类中的`map_action_inverse`、`map_action`和`exploration_noise`方法有关，这些方法分别用于动作的反向映射、动作映射和在动作上添加探索噪声。

**注意**:
- 在使用`collect`函数时，必须确保环境不是异步环境，如果是，请使用`AsyncCollector`。
- 只能指定`n_step`或`n_episode`中的一个，同时指定将导致错误。
- 在训练模式下，建议设置`no_grad`为True以提高性能。
- 如果环境重置时需要特定的配置，请通过`gym_reset_kwargs`正确设置。

**输出示例**:
如果不进行渲染，`collect`函数可能返回如下格式的字典：
```
{
    "n/ep": 5,  // 收集的回合数
    "n/st": 100,  // 收集的步骤数
    "rews": np.array([1.0, 0.5, 2.0, 1.5, 1.0]),  // 各回合的奖励
    "lens": np.array([20, 20, 20, 20, 20]),  // 各回合的长度
    "idxs": np.array([0, 20, 40, 60, 80]),  // 各回合在缓冲区中的起始索引
    "rew": 1.2,  // 平均奖励
    "len": 20,  // 平均回合长度
    "rew_std": 0.5,  // 奖励的标准误
    "len_std": 0  // 回合长度的标准误
}
```
如果进行渲染，除了上述字典外
***
