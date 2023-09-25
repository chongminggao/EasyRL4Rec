python environments/Etsydata/process_data.py

# step 1
python examples/usermodel/run_DeepFM_ensemble.py --env EtsyEnv-v0  --seed 2023 --cuda 6     --epoch 3 --loss "pointneg" --message "pointneg"

# step 2
python examples/policy/run_A2C.py --env EtsyEnv-v0 --seed 2023 --cuda 1 --epoch 50 --num_leave_compute 3 --leave_threshold 0.01 --which_tracker gru --reward_handle cat --window_size 3 --read_message pointneg --message A2C-test_1 &
python examples/policy/run_A2C.py --env EtsyEnv-v0 --seed 2023 --cuda 1 --epoch 50 --num_leave_compute 5 --leave_threshold 0.01 --which_tracker gru --reward_handle cat --window_size 3 --read_message pointneg --message A2C-test_2 &
python examples/policy/run_A2C.py --env EtsyEnv-v0 --seed 2023 --cuda 1 --epoch 50 --num_leave_compute 3 --leave_threshold 0.01 --which_tracker gru --reward_handle cat --window_size 5 --read_message pointneg --message A2C-test_3 &
python examples/policy/run_A2C.py --env EtsyEnv-v0 --seed 2023 --cuda 1 --epoch 50 --num_leave_compute 3 --leave_threshold 0.01 --which_tracker avg --reward_handle cat --window_size 3 --read_message pointneg --message A2C-test_4 &

python examples/advance/run_MOPO.py   --env EtsyEnv-v0  --seed 2023 --cuda 1 --epoch 50  --num_leave_compute 3 --leave_threshold 0.01 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --window_size 3 --read_message "pointneg"  --message "MOPO" &
python examples/advance/run_DORL.py   --env EtsyEnv-v0  --seed 2023 --cuda 1 --epoch 50  --num_leave_compute 3 --leave_threshold 0.01 --which_tracker avg --reward_handle "cat" --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "DORL" &
# python examples/advance/run_Intrinsic.py --env EtsyEnv-v0  --seed 2023 --cuda 1 --epoch 50  --num_leave_compute 3 --leave_threshold 0.01 --which_tracker avg --reward_handle "cat" --lambda_diversity 0.1 --lambda_novelty 0.1 --window_size 3 --read_message "pointneg"  --message "Intrinsic" &
