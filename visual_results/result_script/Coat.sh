# 1. Offline RL(Batch RL) (offpolicy)
python examples/policy/run_DiscreteBCQ.py --env CoatEnv-v0  --seed 2023 --cuda 7 --epoch 100 --num_leave_compute 7 --leave_threshold 6  --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg" --step-per-epoch 1000 --is_exploration_noise --explore_eps 0.4
python examples/policy/run_DiscreteCQL.py --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100 --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg"  --step-per-epoch 1000 --is_exploration_noise
python examples/policy/run_DiscreteCRR.py --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 100 --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg" --step-per-epoch 1000 --is_exploration_noise
python examples/advance/run_SQN.py        --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 100 --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg"  --step-per-epoch 1000 --which_head "shead" --is_exploration_noise

# 2. Online RL with User Model (Model-based or simulation-based RL) 
# 2.1 offpolicy
python examples/policy/run_DQN.py      --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100  --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --is_exploration_noise
python examples/policy/run_C51.py      --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100  --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --is_exploration_noise
python examples/policy/run_DDPG.py --env CoatEnv-v0 --seed 2023 --cuda 4 --epoch 100 --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --is_exploration_noise --remap_eps 0.001 --explore_eps 1.5
python examples/policy/run_TD3.py --env CoatEnv-v0 --seed 2023 --cuda 5 --epoch 100 --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --policy-noise 0.25 --is_exploration_noise --remap_eps 0.001 --explore_eps 1.3

# 2.2 onpolicy
python examples/policy/run_PG.py      --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100  --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --no_exploration_noise
python examples/policy/run_A2C.py     --env CoatEnv-v0  --seed 2023 --cuda 7 --epoch 100  --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --lr 0.005  --ent-coef 0.01 
python examples/policy/run_PPO.py     --env CoatEnv-v0  --seed 2023 --cuda 6 --epoch 100  --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --vf-coef 0.25 --eps-clip 0.3 --no_exploration_noise --lr 0.011
python examples/policy/run_ContinuousPG.py      --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100  --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --no_exploration_noise
python examples/policy/run_ContinuousA2C.py     --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 100  --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --no_exploration_noise
python examples/policy/run_ContinuousPPO.py     --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100  --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --no_exploration_noise

python examples/advance/run_DORL.py        --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 100 --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg" --no_exploration_noise  --lambda_entropy 3 --lr 0.005  --ent-coef 0.01  --no_feature_level
python examples/advance/run_Intrinsic.py        --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 100 --num_leave_compute 7 --leave_threshold 6 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg" --no_exploration_noise --lambda_diversity 0.1  --lambda_novelty 0.05 --lr 0.005  --ent-coef 0.01

