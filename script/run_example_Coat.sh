# run user model
python examples/usermodel/run_DeepFM_ensemble.py --env CoatEnv-v0  --seed 2023 --cuda 1     --epoch 2 --n_models 2 --loss "pointneg" --message "pointneg"
python examples/usermodel/run_DeepFM_IPS.py      --env CoatEnv-v0  --seed 2023 --cuda 0     --epoch 3 --loss "pointneg" --message "DeepFM-IPS"
python examples/usermodel/run_Egreedy.py         --env CoatEnv-v0  --num_leave_compute 4  --leave_threshold 0 --epoch 3 --seed 2023 --cuda 2 --loss "pointneg" --message "epsilon-greedy"
python examples/usermodel/run_LinUCB.py          --env CoatEnv-v0  --num_leave_compute 4  --leave_threshold 0 --epoch 3 --seed 2023 --cuda 3 --loss "pointneg" --message "UCB"

# test state_tracker
python examples/policy/run_A2C.py     --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 10 --which_tracker caser --reward_handle "cat" --window_size 5 --read_message "pointneg"  --message "A2C_caser"
python examples/policy/run_A2C.py     --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 10 --which_tracker gru --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "A2C_gru"
python examples/policy/run_A2C.py     --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 10 --which_tracker sasrec --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "A2C_sasrec"
python examples/policy/run_A2C.py     --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 10 --which_tracker nextitnet --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "A2C_nextitnet"

# run policy
# 1. Offline RL(Batch RL) (offpolicy)
python examples/policy/run_DiscreteCRR.py --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg"  --message "DiscreteCRR-softmax"
python examples/policy/run_DiscreteCQL.py --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat"  --num-quantiles 20 --min-q-weight 10 --window_size 3 --read_message "pointneg"  --message "DiscreteCQL"
python examples/policy/run_DiscreteBCQ.py --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg"  --message "DiscreteBCQ"
# python examples/policy/run_ContinuousBCQ.py --env CoatEnv-v0  --seed 2023 --cuda 3 --epoch 3 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg"  --message "ContinuousBCQ"


# 2. Online RL with User Model (Model-based or simulation-based RL) 
# 2.1 onpolicy
python examples/policy/run_A2C.py           --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "A2C"
python examples/policy/run_A2C.py           --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --remove_recommended_ids --message "A2C-remove"
python examples/policy/run_ContinuousA2C.py --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "ContinuousA2C"
python examples/policy/run_PG.py            --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "PG"
python examples/policy/run_ContinuousPG.py  --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "ContinuousPG"
python examples/policy/run_PPO.py           --env CoatEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "PPO"
python examples/policy/run_ContinuousPPO.py --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "ContinuousPPO"

# 2.2 offpolicy
python examples/policy/run_DQN.py     --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "DQN"
python examples/policy/run_DQN.py     --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --remove_recommended_ids --message "DQN-remove"
python examples/policy/run_DQN.py     --env CoatEnv-v0  --seed 2023 --cuda 1 --step-per-collect 200 --step-per-epoch 10000 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "DQN-1w-200"
python examples/policy/run_DQN.py     --env CoatEnv-v0  --seed 2023 --cuda 2 --step-per-collect 200 --step-per-epoch 20000 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "DQN-2w-200"
python examples/policy/run_QRDQN.py   --env CoatEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "QRDQN"
python examples/policy/run_C51.py     --env CoatEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --remove_recommended_ids --message "C51-remove"
python examples/policy/run_DDPG.py    --env CoatEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "DDPG"
python examples/policy/run_TD3.py     --env CoatEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "TD3"

# run advance
python examples/advance/run_A2C_IPS.py    --env CoatEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "DeepFM-IPS"  --message "IPS"
python examples/advance/run_SQN.py        --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg"  --message "SQN"
python examples/advance/run_MOPO.py       --env CoatEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --window_size 3 --read_message "pointneg"  --message "MOPO"
python examples/advance/run_DORL.py       --env CoatEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "DORL"
python examples/advance/run_Intrinsic.py  --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 3 --which_tracker avg --reward_handle "cat" --lambda_diversity 0.1 --lambda_novelty 0.1 --window_size 3 --read_message "pointneg"  --message "Intrinsic"
