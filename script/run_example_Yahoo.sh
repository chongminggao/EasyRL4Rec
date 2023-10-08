# run user model
python examples/usermodel/run_DeepFM_ensemble.py --env YahooEnv-v0  --seed 2023 --cuda 1     --epoch 2 --n_models 2 --loss "pointneg" --message "pointneg"

# test state_tracker


# run policy
# 1. Offline RL(Batch RL) (offpolicy)


# 2. Online RL with User Model (Model-based or simulation-based RL) 
# 2.1 onpolicy
python examples/policy/run_A2C.py     --env YahooEnv-v0  --seed 2023 --cuda 1 --epoch 2  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "A2C-new"
