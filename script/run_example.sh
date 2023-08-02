# run user model
python examples/usermodel/run_DeepFM_ensemble.py --env KuaiEnv-v0  --seed 2023 --cuda 0 --epoch 1 --loss "pointneg" --message "pointneg"
python examples/usermodel/run_DeepFM_IPS.py      --env KuaiEnv-v0  --seed 2023 --cuda 1 --epoch 1 --loss "pointneg" --message "DeepFM-IPS"
python examples/usermodel/run_Egreedy.py         --env KuaiEnv-v0  --num_leave_compute 4  --leave_threshold 0 --epoch 1 --seed 2023 --cuda 2 --loss "pointneg" --message "epsilon-greedy"
python examples/usermodel/run_LinUCB.py          --env KuaiEnv-v0  --num_leave_compute 4  --leave_threshold 0 --epoch 1 --seed 2023 --cuda 3 --loss "pointneg" --message "UCB"

# run policy
python examples/policy/run_SQN.py --env KuaiEnv-v0  --seed 2023 --cuda 0 --epoch 3  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg"  --message "SQN"
python examples/policy/run_CRR.py --env KuaiEnv-v0  --seed 2023 --cuda 0 --epoch 3  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg"  --message "CRR"
python examples/policy/run_CQL.py --env KuaiEnv-v0  --seed 2023 --cuda 1 --epoch 3  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat"  --num-quantiles 20 --min-q-weight 10 --window_size 3 --read_message "pointneg"  --message "CQL"
python examples/policy/run_BCQ.py --env KuaiEnv-v0  --seed 2023 --cuda 0 --epoch 3  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg"  --message "BCQ"

python examples/policy/run_A2C_IPS.py --env KuaiEnv-v0  --seed 2023 --cuda 1 --epoch 3  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0    --lambda_entropy 0    --window_size 3 --read_message "DeepFM-IPS"  --message "IPS"
python examples/policy/run_A2C.py     --env KuaiEnv-v0  --seed 2023 --cuda 0 --epoch 3  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0    --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MBPO"
python examples/policy/run_A2C.py     --env KuaiEnv-v0  --seed 2023 --cuda 1 --epoch 3  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 0    --window_size 3 --read_message "pointneg"  --message "MOPO"
python examples/policy/run_A2C.py     --env KuaiEnv-v0  --seed 2023 --cuda 0 --epoch 3  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --lambda_entropy 5    --window_size 3 --read_message "pointneg"  --message "DORL"

