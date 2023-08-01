# run user model
python examples/usermodel/run_DeepFM_ensemble.py --env KuaiEnv-v0  --seed 2023 --cuda 0 --epoch 1 --loss "pointneg" --message "pointneg"
python examples/usermodel/run_DeepFM_IPS.py      --env KuaiEnv-v0  --seed 2023 --cuda 1 --epoch 1 --loss "pointneg" --message "DeepFM-IPS"
python examples/usermodel/run_Egreedy.py         --env KuaiEnv-v0  --num_leave_compute 4  --leave_threshold 0 --epoch 1 --seed 2023 --cuda 2 --loss "pointneg" --message "epsilon-greedy"
python examples/usermodel/run_LinUCB.py          --env KuaiEnv-v0  --num_leave_compute 4  --leave_threshold 0 --epoch 1 --seed 2023 --cuda 3 --loss "pointneg" --message "UCB"