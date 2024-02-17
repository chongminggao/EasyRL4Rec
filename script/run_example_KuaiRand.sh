# Run User Model
python examples/usermodel/run_DeepFM_ensemble.py --env KuaiRand-v0  --seed 2023 --cuda 6 --epoch 5 --n_models 5 --loss "pointneg" --message "pointneg"
python examples/usermodel/run_DeepFM_IPS.py      --env KuaiRand-v0  --seed 2023 --cuda 1 --epoch 5 --loss "pointneg" --message "DeepFM-IPS"


# Run Policy
# 1. Offline RL (Batch RL)
python examples/policy/run_DiscreteBCQ.py --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3 --unlikely-action-threshold 0.2 --explore_eps 0.4 --read_message "pointneg"  --message "BCQ"
python examples/policy/run_DiscreteCQL.py --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3 --min-q-weight 0.3 --explore_eps 0.4 --read_message "pointneg"  --message "CQL"
python examples/policy/run_DiscreteCRR.py --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3 --explore_eps 0.01 --read_message "pointneg"  --message "CRR"
python examples/advance/run_SQN.py        --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3 --unlikely-action-threshold 0.6 --explore_eps 0.4 --read_message "pointneg"  --message "SQN"

# 2. Online RL with User Model
# 2.1 offpolicy
python examples/policy/run_DQN.py  --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3 --target-update-freq 80 --explore_eps 0.001 --read_message "pointneg"  --message "DQN"
python examples/policy/run_C51.py  --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3 --v-min 0. --v-max 1. --explore_eps 0.005 --read_message "pointneg"  --message "C51"
python examples/policy/run_DDPG.py --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3 --remap 0.001 --explore_eps 1.2 --read_message "pointneg"  --message "DDPG"    
python examples/policy/run_TD3.py  --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3 --remap 0.001 --explore_eps 1.5 --read_message "pointneg"  --message "TD3" 

# 2.2 onpolicy
python examples/policy/run_PG.py            --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3  --read_message "pointneg"  --message "PG"
python examples/policy/run_A2C.py           --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3  --read_message "pointneg"  --message "A2C"
python examples/policy/run_PPO.py           --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3  --vf-coef 0.25 --max-grad-norm 0.2 --read_message "pointneg"  --message "PPO"
python examples/policy/run_ContinuousPG.py  --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3  --lr 0.002 --remap_eps 0.002 --read_message "pointneg"  --message "PG(C)"
python examples/policy/run_ContinuousA2C.py --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3  --read_message "pointneg"  --message "A2C(C)"
python examples/policy/run_ContinuousPPO.py --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3  --read_message "pointneg"  --message "PPO(C)"

python examples/advance/run_DORL.py         --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3  --read_message "pointneg"  --message "DORL"
python examples/advance/run_Intrinsic.py    --env KuaiRand-v0  --seed 2023 --cuda 0 --which_tracker avg --reward_handle "cat" --window_size 3  --step-per-epoch 30000 --lambda_diversity 0.005 --lambda_novelty 0.001 --read_message "pointneg"  --message "Intrinsic"


# Others
python examples/advance/run_MOPO.py --env KuaiRand-v0  --seed 2023 --cuda 3 --epoch 10 --which_tracker avg --reward_handle "cat" --lambda_variance 0.05 --window_size 3 --read_message "pointneg"  --message "MOPO"
python examples/advance/run_CIRS.py --env KuaiRand-v0  --seed 2023 --cuda 0 --epoch 10 --which_tracker avg --reward_handle "cat" --tau 100 --window_size 3 --read_message "CIRS_UM"  --message "CIRS"