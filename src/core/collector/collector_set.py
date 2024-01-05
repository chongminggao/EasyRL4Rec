
# @Author  : Chongming GAO
# @FileName: collector_set.py

from typing import Callable, Optional, Dict, Any, Union, List
from src.tianshou.tianshou.data import Batch, VectorReplayBuffer, ReplayBuffer

import sys
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from src.core.collector.collector import Collector

# class Exec_Each_Collector(object):
#     def __init__(self, func):
#         self.func = func
#
#     def __get__(self, instance, owner, *args, **kwargs):
#         for collector in instance.collector_list:
#             collector_func = getattr(collector, self.func.__name__)
#             collector_func(*args, **kwargs)
#         return partial(self.func, self=instance)

class CollectorSet(object):
    def __init__(self, policy, envs_dict, buffer_size, env_num,
                 preprocess_fn: Optional[Callable[..., Batch]] = None, exploration_noise: bool = False,
                 force_length=10):
        self.collector_dict = {}

        remove_recommended_ids_dict = {"FB": False, "NX_0": True, f"NX_{force_length}": True}
        force_length_dict = {"FB": 0, "NX_0": 0, f"NX_{force_length}": force_length}

        for name, envs in envs_dict.items():
            collector = Collector(
                policy, envs,
                VectorReplayBuffer(buffer_size, env_num),
                preprocess_fn=preprocess_fn,
                exploration_noise=exploration_noise if name == "FB" else False,
                remove_recommended_ids=remove_recommended_ids_dict[name],
                force_length=force_length_dict[name]
            )
            self.collector_dict[name] = collector

        self.env = envs_dict["FB"]
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self.exploration_noise = exploration_noise
        self.env_num = env_num

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        for name, collector in self.collector_dict.items():
            collector._assign_buffer(buffer)

    def reset_stat(self) -> None:
        for name, collector in self.collector_dict.items():
            collector.reset_stat()

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        for name, collector in self.collector_dict.items():
            collector.reset_buffer(keep_statistics)

    def reset_env(self) -> None:
        """Reset all environments."""
        for name, collector in self.collector_dict.items():
            collector.reset_env()

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        for name, collector in self.collector_dict.items():
            collector._reset_state(id)

    def collect(
            self,
            n_step: Optional[int] = None,
            n_episode: Optional[int] = None,
            random: bool = False,
            render: Optional[float] = None,
            no_grad: bool = True,
    ) -> Dict[str, Any]:
        all_res = {}
        for name, collector in self.collector_dict.items():
            res = collector.collect(n_step, n_episode, random, render, no_grad, is_train=False)
            res_k = {name + "_" + k: v for k, v in res.items()} if name != "FB" else res
            all_res.update(res_k)
        self.collect_step = self.collector_dict["FB"].collect_step
        self.collect_episode = self.collector_dict["FB"].collect_episode
        self.collect_time = self.collector_dict["FB"].collect_time
        return all_res
