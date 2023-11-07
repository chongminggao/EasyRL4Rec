import argparse

def get_env_args(args):
    env = args.env

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')

    parser.add_argument('--is_binarize', dest='is_binarize', action='store_true')
    parser.add_argument('--no_binarize', dest='is_binarize', action='store_false')

    parser.add_argument('--is_need_transform', dest='need_transform', action='store_true')
    parser.add_argument('--no_need_transform', dest='need_transform', action='store_false')


    if env == "CoatEnv-v0":
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=10, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "YahooEnv-v0":
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=120, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "MovieLensEnv-v0":
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=120, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "KuaiRand-v0":
        parser.set_defaults(is_userinfo=False)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = False
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--rating_threshold", type=float, default=1)
        parser.add_argument("--yfeat", type=str, default="is_click")

        parser.add_argument('--leave_threshold', default=0, type=float)
        parser.add_argument('--num_leave_compute', default=10, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "KuaiEnv-v0":
        parser.set_defaults(is_userinfo=False)
        parser.set_defaults(is_binarize=False)
        parser.set_defaults(need_transform=True)
        # args.entropy_on_user = False
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--yfeat", type=str, default="watch_ratio_normed")

        # parser.add_argument('--leave_threshold', default=1, type=float)
        # parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--leave_threshold', default=0, type=float)
        parser.add_argument('--num_leave_compute', default=10, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    parser.add_argument('--force_length', type=int, default=10)
    parser.add_argument("--top_rate", type=float, default=0.8)

    args_new = parser.parse_known_args()[0]
    args.__dict__.update(args_new.__dict__)
    if env == "KuaiEnv-v0":
        args.use_userEmbedding = False

    return args


def get_true_env(args, read_user_num=None):
    if args.env == "CoatEnv-v0":
        from environments.coat.CoatEnv import CoatEnv
        from environments.coat.CoatData import CoatData
        mat, df_item, mat_distance = CoatEnv.load_env_data()
        kwargs_um = {"mat": mat,
                     "df_item": df_item,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}
        env = CoatEnv(**kwargs_um)
        dataset = CoatData()
    elif args.env == "YahooEnv-v0":
        from environments.YahooR3.YahooEnv import YahooEnv
        from environments.YahooR3.YahooData import YahooData
        mat, mat_distance = YahooEnv.load_env_data()
        kwargs_um = {"mat": mat,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}

        env = YahooEnv(**kwargs_um)
        dataset = YahooData()
    elif args.env == "MovieLensEnv-v0":
        from environments.MovieLens.MovieLensEnv import MovieLensEnv
        from environments.MovieLens.MovieLensData import MovieLensData
        mat, mat_distance = MovieLensEnv.load_env_data()
        kwargs_um = {"mat": mat,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}

        env = MovieLensEnv(**kwargs_um)
        dataset = MovieLensData()
    elif args.env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.KuaiRandEnv import KuaiRandEnv
        from environments.KuaiRand_Pure.KuaiRandData import KuaiRandData
        mat, list_feat, mat_distance = KuaiRandEnv.load_env_data(args.yfeat, read_user_num=read_user_num)
        kwargs_um = {"yname": args.yfeat,
                     "mat": mat,
                     "mat_distance": mat_distance,
                     "list_feat": list_feat,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}
        env = KuaiRandEnv(**kwargs_um)
        dataset = KuaiRandData()
    elif args.env == "KuaiEnv-v0":
        from environments.KuaiRec.KuaiEnv import KuaiEnv
        from environments.KuaiRec.KuaiData import KuaiData
        mat, lbe_user, lbe_item, list_feat, df_dist_small = KuaiEnv.load_env_data()
        kwargs_um = {"mat": mat,
                     "lbe_user": lbe_user,
                     "lbe_item": lbe_item,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init,
                     "list_feat": list_feat,
                     "df_dist_small": df_dist_small}
        env = KuaiEnv(**kwargs_um)
        dataset = KuaiData()
    return env, dataset, kwargs_um