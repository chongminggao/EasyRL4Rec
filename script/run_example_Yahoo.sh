# Run User Model
python examples/usermodel/run_DeepFM_ensemble.py --env YahooEnv-v0  --seed 2023 --cuda 1 --epoch 5 --n_models 5 --loss "pointneg" --message "pointneg" 
python examples/usermodel/run_DeepFM_IPS.py      --env YahooEnv-v0  --seed 2023 --cuda 0 --epoch 5 --loss "pointneg" --message "DeepFM-IPS" 
# python examples/usermodel/run_Egreedy.py         --env YahooEnv-v0  --seed 2023 --cuda 2 --epoch 5 --loss "point" --message "epsilon-greedy-point" 
# python examples/usermodel/run_LinUCB.py          --env YahooEnv-v0  --seed 2023 --cuda 3 --epoch 5 --loss "point" --message "UCB-point" 

# Run Policy
# 1. Offline RL (Batch RL)
python examples/policy/run_DiscreteBCQ.py --env YahooEnv-v0  --seed 2023 --cuda 0 --epoch 100  --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --step-per-epoch 1000 --is_exploration_noise --explore_eps 0.4 --read_message "pointneg"  --message "BCQ"
python examples/policy/run_DiscreteCQL.py --env YahooEnv-v0  --seed 2023 --cuda 1 --epoch 100  --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg"  --step-per-epoch 1000 --is_exploration_noise --read_message "pointneg"  --message "CQL"
python examples/policy/run_DiscreteCRR.py --env YahooEnv-v0  --seed 2023 --cuda 0 --epoch 100  --which_tracker avg --reward_handle "cat"  --window_size 3 --step-per-epoch 1000 --is_exploration_noise --read_message "pointneg"  --message "CRR"
python examples/advance/run_SQN.py        --env YahooEnv-v0  --seed 2023 --cuda 1 --epoch 100  --which_tracker avg --reward_handle "cat"  --window_size 3 --step-per-epoch 1000 --which_head "shead" --is_exploration_noise --read_message "pointneg"  --message "SQN"

# 2. Online RL with User Model
# 2.1 offpolicy
python examples/policy/run_DQN.py   --env YahooEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --is_exploration_noise --read_message "pointneg" --message "DQN"
python examples/policy/run_C51.py   --env YahooEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --is_exploration_noise --read_message "pointneg" --message "C51"
python examples/policy/run_DDPG.py  --env YahooEnv-v0  --seed 2023 --cuda 4 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --is_exploration_noise --remap_eps 0.001 --explore_eps 1.5 --read_message "pointneg" --message "DDPG"
python examples/policy/run_TD3.py   --env YahooEnv-v0  --seed 2023 --cuda 5 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --policy-noise 0.25 --is_exploration_noise --remap_eps 0.001 --explore_eps 1.3 --read_message "pointneg" --message "TD3"
python examples/policy/run_QRDQN.py --env YahooEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --is_exploration_noise --read_message "pointneg" --message "QRDQN" 

# 2.2 onpolicy
python examples/policy/run_PG.py            --env YahooEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --no_exploration_noise --read_message "pointneg" --message "PG"
python examples/policy/run_A2C.py           --env YahooEnv-v0  --seed 2023 --cuda 7 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --lr 0.005  --ent-coef 0.01 --read_message "pointneg" --message "A2C"
python examples/policy/run_PPO.py           --env YahooEnv-v0  --seed 2023 --cuda 6 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --vf-coef 0.25 --eps-clip 0.3 --no_exploration_noise --lr 0.011 --read_message "pointneg" --message "PPO"
python examples/policy/run_ContinuousPG.py  --env YahooEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --no_exploration_noise --read_message "pointneg" --message "ContinuousPG"
python examples/policy/run_ContinuousA2C.py --env YahooEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --no_exploration_noise --read_message "pointneg" --message "ContinuousA2C"
python examples/policy/run_ContinuousPPO.py --env YahooEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --no_exploration_noise --read_message "pointneg" --message "ContinuousPPO"

python examples/advance/run_DORL.py         --env YahooEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --no_exploration_noise --lambda_entropy 3 --lr 0.005  --ent-coef 0.01  --no_feature_level --read_message "pointneg" --message "DORL"
python examples/advance/run_Intrinsic.py    --env YahooEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --no_exploration_noise --lambda_diversity 0.1  --lambda_novelty 0.05 --lr 0.005  --ent-coef 0.01 --read_message "pointneg" --message "Intrinsic"


# Others
python examples/advance/run_A2C_IPS.py    --env YahooEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "DeepFM-IPS"  --message "IPS"
python examples/advance/run_MOPO.py       --env YahooEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --window_size 3 --read_message "pointneg"  --message "MOPO"