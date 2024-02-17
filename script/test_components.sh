# State Tracker
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --epoch 100 --which_tracker avg --read_message "pointneg"  --message "PG_avg"
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --epoch 100 --which_tracker caser --window_size 5 --read_message "pointneg"  --message "PG_caser"
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --epoch 100 --which_tracker gru --read_message "pointneg"  --message "PG_gru"
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --epoch 100 --which_tracker sasrec --num_heads 2 --read_message "pointneg"  --message "PG_sasrec"
python examples/policy/run_PG.py --env KuaiEnv-v0 --cuda 0 --epoch 100 --which_tracker nextitnet --read_message "pointneg"  --message "PG_nextitnet"

# Construction methods
python examples/policy/run_DiscreteCRR.py --env KuaiEnv-v0  --cuda 0 --epoch 100 --explore_eps 0.01 --construction_method normal --read_message "pointneg"  --message "CRR"
python examples/policy/run_DiscreteCRR.py --env KuaiEnv-v0  --cuda 0 --epoch 100 --explore_eps 0.01 --construction_method counterfactual --read_message "pointneg"  --message "CRR_counterfactual"
python examples/policy/run_DiscreteCRR.py --env KuaiEnv-v0  --cuda 0 --epoch 100 --explore_eps 0.01 --construction_method convolution --read_message "pointneg"  --message "CRR_convolution"
