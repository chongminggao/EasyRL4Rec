
# 1. Offline RL(Batch RL) (offpolicy)
python examples/policy/run_DiscreteBCQ.py --env KuaiEnv-v0  --cuda 0 --unlikely-action-threshold 0.2 --explore_eps 0.4 --read_message "pointneg"  --message "BCQ"
python examples/policy/run_DiscreteCQL.py --env KuaiEnv-v0  --cuda 0 --min-q-weight 0.3 --explore_eps 0.4 --read_message "pointneg"  --message "CQL"
python examples/policy/run_DiscreteCRR.py --env KuaiEnv-v0  --cuda 0 --explore_eps 0.01 --read_message "pointneg"  --message "CRR"
python examples/advance/run_SQN.py --env KuaiEnv-v0  --cuda 0 --unlikely-action-threshold 0.6 --explore_eps 0.4 --read_message "pointneg"  --message "SQN"

# 2. Online RL with User Model (Model-based or simulation-based RL)
# 2.1 offpolicy
python examples/policy/run_DQN.py --env KuaiEnv-v0  --cuda 0 --target-update-freq 80 --explore_eps 0.001 --read_message "pointneg"  --message "DQN"
python examples/policy/run_C51.py --env KuaiEnv-v0  --cuda 0 --v-min 0. --v-max 1. --explore_eps 0.005 --read_message "pointneg"  --message "C51"
python examples/policy/run_DDPG.py --env KuaiEnv-v0  --cuda 0 --remap 0.001 --explore_eps 1.2 --read_message "pointneg"  --message "DDPG"    
python examples/policy/run_TD3.py --env KuaiEnv-v0  --cuda 0 --remap 0.001 --explore_eps 1.5 --read_message "pointneg"  --message "TD3" 
# 2.2 onpolicy
python examples/policy/run_PG.py --env KuaiEnv-v0  --cuda 0 --read_message "pointneg"  --message "PG"
python examples/policy/run_A2C.py --env KuaiEnv-v0  --cuda 0 --read_message "pointneg"  --message "A2C"
python examples/policy/run_PPO.py --env KuaiEnv-v0  --cuda 0 --vf-coef 0.25 --max-grad-norm 0.2 --read_message "pointneg"  --message "PPO"
python examples/policy/run_ContinuousPG.py --env KuaiEnv-v0  --cuda 0 --lr 0.002 --remap_eps 0.002 --read_message "pointneg"  --message "PG(C)"
python examples/policy/run_ContinuousA2C.py --env KuaiEnv-v0  --cuda 0 --read_message "pointneg"  --message "A2C(C)"
python examples/policy/run_ContinuousPPO.py --env KuaiEnv-v0  --cuda 0 --read_message "pointneg"  --message "PPO(C)"
python examples/advance/run_DORL.py --env KuaiEnv-v0  --cuda 0 --read_message "pointneg"  --message "DORL"
python examples/advance/run_Intrinsic.py --env KuaiEnv-v0  --cuda 0 --step-per-epoch 30000 --lambda_diversity 0.005 --lambda_novelty 0.001 --read_message "pointneg"  --message "Intrinsic"

# 3. State Tracker
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --which_tracker avg --read_message "pointneg"  --message "PG_avg"
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --which_tracker caser --window_size 5 --read_message "pointneg"  --message "PG_caser"
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --which_tracker gru --read_message "pointneg"  --message "PG_gru"
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --which_tracker sasrec --num_heads 2 --read_message "pointneg"  --message "PG_sasrec"
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --which_tracker nextitnet --read_message "pointneg"  --message "PG_nextitnet"

# 4. Construction methods
python examples/policy/run_DiscreteCRR.py --env KuaiEnv-v0  --cuda 0 --explore_eps 0.01 --no_offline_counterfactual_permutate --read_message "pointneg"  --message "CRR"
python examples/policy/run_DiscreteCRR.py --env KuaiEnv-v0  --cuda 0 --explore_eps 0.01 --is_offline_counterfactual_permutate --offline_repeat_num 10 --read_message "pointneg"  --message "CRR_counterfactual"
