# Run User Model
python examples/usermodel/run_DeepFM_ensemble.py --env MovieLensEnv-v0  --seed 2023 --cuda 1  --epoch 5 --n_models 5 --loss "pointneg" --message "pointneg"
python examples/usermodel/run_DeepFM_IPS.py      --env MovieLensEnv-v0  --seed 2023 --cuda 0  --epoch 5 --loss "pointneg" --message "DeepFM-IPS"
# python examples/usermodel/run_DeepFM_ensemble.py --env MovieLensEnv-v0  --seed 2023 --cuda 1  --epoch 5 --tau 100 --n_models 1 --loss "pointneg" --message "CIRS_UM"
# python examples/usermodel/run_Egreedy.py         --env MovieLensEnv-v0  --seed 2023 --cuda 2  --epoch 5 --loss "pointneg" --message "epsilon-greedy"
# python examples/usermodel/run_LinUCB.py          --env MovieLensEnv-v0  --seed 2023 --cuda 3  --epoch 5 --loss "pointneg" --message "UCB"

# Run Policy
# 1. Offline RL(Batch RL)
python examples/policy/run_DiscreteCRR.py --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg3"  --message "CRR"
python examples/policy/run_DiscreteCQL.py --env MovieLensEnv-v0  --seed 2023 --cuda 5 --epoch 100 --explore_eps 0.5 --which_tracker avg --reward_handle "cat"  --num-quantiles 20 --min-q-weight 10 --window_size 3 --read_message "pointneg3"  --message "CQL"
python examples/policy/run_DiscreteBCQ.py --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --explore_eps 0.5 --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg3"  --message "BCQ"
python examples/advance/run_SQN.py        --env MovieLensEnv-v0  --seed 2023 --cuda 7 --epoch 100 --step-per-epoch 10000 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg3"  --message "SQN"

# 2. Online RL with User Model
# 2.1 offpolicy
python examples/policy/run_DQN.py     --env MovieLensEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg3"  --message "DQN"
python examples/policy/run_C51.py     --env MovieLensEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg3"  --message "C51"
python examples/policy/run_DDPG.py    --env MovieLensEnv-v0  --seed 2023 --cuda 2 --epoch 100 --step-per-epoch 30000 --explore_eps 1.0 --remap_eps 0.001 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg3"  --message "DDPG"
python examples/policy/run_TD3.py     --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --explore_eps 0.9 --remap_eps 0.9 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg3"  --message "TD3"

# 2.2 onpolicy
python examples/policy/run_PG.py            --env MovieLensEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg3"  --message "PG"
python examples/policy/run_A2C.py           --env MovieLensEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg_1"  --message "A2C"
python examples/policy/run_PPO.py           --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg3"  --message "PPO"
python examples/policy/run_ContinuousPG.py  --env MovieLensEnv-v0  --seed 2023 --cuda 0 --epoch 100 --remap_eps 0 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg3"  --message "PG(C)"
python examples/policy/run_ContinuousA2C.py --env MovieLensEnv-v0  --seed 2023 --cuda 6 --epoch 100 --remap_eps 0 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg3"  --message "A2C(C)"
python examples/policy/run_ContinuousPPO.py --env MovieLensEnv-v0  --seed 2023 --cuda 6 --epoch 100 --remap_eps 0 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg3"  --message "PPO(C)"

python examples/advance/run_DORL.py         --env MovieLensEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --lambda_entropy 5     --window_size 3 --read_message "pointneg3"  --message "DORL"
python examples/advance/run_Intrinsic.py    --env MovieLensEnv-v0  --seed 2023 --cuda 6 --epoch 100 --step-per-epoch 50000 --which_tracker avg --reward_handle "cat" --lambda_diversity 0.2 --lambda_novelty 0.2 --window_size 3 --read_message "pointneg3"  --message "Intrinsic"


# Others
python examples/advance/run_A2C_IPS.py --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "DeepFM-IPS"  --message "IPS"
python examples/advance/run_MOPO.py    --env MovieLensEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --window_size 3 --read_message "pointneg"  --message "MOPO"
python examples/advance/run_CIRS.py    --env MovieLensEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat" --tau 100 --window_size 3 --read_message "CIRS_UM"  --message "CIRS"
