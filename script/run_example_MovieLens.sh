# run user model

python examples/usermodel/run_DeepFM_ensemble.py --env MovieLensEnv-v0  --seed 2023 --cuda 1  --epoch 5 --n_models 3 --loss "point" --message "point" &
python examples/usermodel/run_DeepFM_IPS.py      --env MovieLensEnv-v0  --seed 2023 --cuda 0  --epoch 5 --loss "point" --message "DeepFM-IPS-point" &
python examples/usermodel/run_Egreedy.py         --env MovieLensEnv-v0  --seed 2023 --cuda 2  --epoch 5 --loss "point" --message "epsilon-greedy-point" &
python examples/usermodel/run_LinUCB.py          --env MovieLensEnv-v0  --seed 2023 --cuda 3  --epoch 5 --loss "point" --message "UCB-point" &

python examples/usermodel/run_DeepFM_ensemble.py --env MovieLensEnv-v0  --seed 2023 --cuda 1  --epoch 5 --n_models 3 --loss "pointneg" --message "pointneg" &
#python examples/usermodel/run_DeepFM_IPS.py      --env MovieLensEnv-v0  --seed 2023 --cuda 0  --epoch 5 --loss "pointneg" --message "DeepFM-IPS" &
#python examples/usermodel/run_Egreedy.py         --env MovieLensEnv-v0  --seed 2023 --cuda 2  --epoch 5 --loss "pointneg" --message "epsilon-greedy" &
#python examples/usermodel/run_LinUCB.py          --env MovieLensEnv-v0  --seed 2023 --cuda 3  --epoch 5 --loss "pointneg" --message "UCB" &

# test state_tracker
python examples/policy/run_A2C.py     --env MovieLensEnv-v0  --seed 2023 --cuda 1 --epoch 10 --which_tracker caser --reward_handle "cat" --window_size 5 --read_message "point"  --message "A2C_caser"
python examples/policy/run_A2C.py     --env MovieLensEnv-v0  --seed 2023 --cuda 1 --epoch 10 --which_tracker gru --reward_handle "cat" --window_size 3 --read_message "point"  --message "A2C_gru"
python examples/policy/run_A2C.py     --env MovieLensEnv-v0  --seed 2023 --cuda 1 --epoch 10 --which_tracker sasrec --reward_handle "cat" --window_size 3 --read_message "point"  --message "A2C_sasrec"
python examples/policy/run_A2C.py     --env MovieLensEnv-v0  --seed 2023 --cuda 1 --epoch 10 --which_tracker nextitnet --reward_handle "cat" --window_size 3 --read_message "point"  --message "A2C_nextitnet"

# run policy
# 1. Offline RL(Batch RL) (offpolicy)
python examples/policy/run_DiscreteCRR.py --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "point"  --message "DiscreteCRR"
python examples/policy/run_DiscreteCQL.py --env MovieLensEnv-v0  --seed 2023 --cuda 5 --epoch 100 --which_tracker avg --reward_handle "cat"  --num-quantiles 20 --min-q-weight 10 --window_size 3 --read_message "point"  --message "DiscreteCQL"
python examples/policy/run_DiscreteBCQ.py --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "point"  --message "DiscreteBCQ"

# 2. Online RL with User Model (Model-based or simulation-based RL) 
# 2.1 onpolicy
python examples/policy/run_A2C.py           --env MovieLensEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "A2C"
python examples/policy/run_PG.py            --env MovieLensEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "PG"
python examples/policy/run_ContinuousPG.py  --env MovieLensEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "ContinuousPG"
python examples/policy/run_DiscretePPO.py   --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "DiscretePPO"
python examples/policy/run_ContinuousPPO.py --env MovieLensEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "ContinuousPPO"

# 2.2 offpolicy
python examples/policy/run_DQN.py     --env MovieLensEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "DQN"
python examples/policy/run_QRDQN.py   --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "QRDQN"
python examples/policy/run_C51.py     --env MovieLensEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "C51"
python examples/policy/run_DDPG.py    --env MovieLensEnv-v0  --seed 2023 --cuda 5 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "DDPG"
python examples/policy/run_TD3.py     --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "point"  --message "TD3"

# run advance
python examples/advance/run_A2C_IPS.py    --env MovieLensEnv-v0  --seed 2023 --cuda 3 --epoch 100 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "DeepFM-IPS"  --message "IPS"
python examples/advance/run_SQN.py        --env MovieLensEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "point"  --message "SQN"
python examples/advance/run_MOPO.py       --env MovieLensEnv-v0  --seed 2023 --cuda 2 --epoch 100 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --window_size 3 --read_message "point"  --message "MOPO"
python examples/advance/run_DORL.py       --env MovieLensEnv-v0  --seed 2023 --cuda 1 --epoch 100 --which_tracker avg --reward_handle "cat" --lambda_entropy 5     --window_size 3 --read_message "point"  --message "DORL"
python examples/advance/run_Intrinsic.py  --env MovieLensEnv-v0  --seed 2023 --cuda 0 --epoch 100 --which_tracker avg --reward_handle "cat" --lambda_diversity 0.1 --lambda_novelty 0.1 --window_size 3 --read_message "point"  --message "Intrinsic"
