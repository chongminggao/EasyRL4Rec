import argparse


def get_features(env, is_userinfo=False):
    if env == "CoatEnv-v0":
        user_features = ["user_id", 'gender_u', 'age', 'location', 'fashioninterest']
        item_features = ['item_id', 'gender_i', "jackettype", 'color', 'onfrontpage']
        reward_features = ["rating"]
    elif env == "KuaiRand-v0":
        user_features = ["user_id", 'user_active_degree', 'is_live_streamer', 'is_video_author',
                         'follow_user_num_range',
                         'fans_user_num_range', 'friend_user_num_range', 'register_days_range'] \
                        + [f'onehot_feat{x}' for x in range(18)]
        if not is_userinfo:
            user_features = ["user_id"]
        item_features = ["item_id"] + ["feat" + str(i) for i in range(3)] + ["duration_normed"]
        reward_features = ["is_click"]
    elif env == "KuaiEnv-v0":
        user_features = ["user_id"]
        item_features = ["item_id"] + ["feat" + str(i) for i in range(4)] + ["duration_normed"]
        reward_features = ["watch_ratio_normed"]
    elif env == "YahooEnv-v0":
        user_features = ["user_id"]
        item_features = ['item_id']
        reward_features = ["rating"]
    elif env == 'EtsyEnv-v0':
        user_features = ["user_id"]
        item_features = ['item_id'] + ["feat" + str(i) for i in range(2)]
        reward_features = ["rating"]


    return user_features, item_features, reward_features

def get_training_data(env):
    df_train, df_user, df_item, list_feat = None, None, None, None
    if env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        df_train, df_user, df_item, list_feat = CoatEnv.get_df_coat("train.ascii")
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        df_train, df_user, df_item, list_feat = KuaiRandEnv.get_df_kuairand("train_processed.csv")
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        df_train, df_user, df_item, list_feat = KuaiEnv.get_df_kuairec("big_matrix_processed.csv")
    elif env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        df_train, df_user, df_item, list_feat = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-train.txt")
    elif env == "EtsyEnv-v0":
        from environments.Etsydata.etsy import EtsyEnv
        df_train, df_user, df_item, list_feat = EtsyEnv.get_df_etsy("df_train.csv")

    return df_train, df_user, df_item, list_feat

def get_training_item_domination(env):
    item_feat_domination = None
    if env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        item_feat_domination = CoatEnv.get_domination()
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        item_feat_domination = KuaiRandEnv.get_domination()
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        item_feat_domination = KuaiEnv.get_domination()
    elif env == "YahooEnv-v0":
        item_feat_domination = None
    elif env == "EtsyEnv-v0":
        from environments.Etsydata.etsy import EtsyEnv
        item_feat_domination = EtsyEnv.get_domination()

    return item_feat_domination

def get_item_similarity(env):
    item_similarity = None
    if env == "CoatEnv-v0":    
        from environments.coat.env.Coat import CoatEnv    
        item_similarity = CoatEnv.get_item_similarity()
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        item_similarity = KuaiRandEnv.get_item_similarity()
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        item_similarity = KuaiEnv.get_item_similarity()
    elif env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        item_similarity = YahooEnv.get_item_similarity()
    elif env == "EtsyEnv-v0":
        from environments.Etsydata.etsy import EtsyEnv
        item_similarity = EtsyEnv.get_item_similarity()

    return item_similarity

def get_item_popularity(env):
    item_popularity = None
    if env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv   
        item_popularity = CoatEnv.get_item_popularity()
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        item_popularity = KuaiRandEnv.get_item_popularity()
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        item_popularity = KuaiEnv.get_item_popularity()
    elif env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        item_popularity = YahooEnv.get_item_popularity()
    elif env == "EtsyEnv-v0":
        from environments.Etsydata.etsy import EtsyEnv
        item_popularity = EtsyEnv.get_item_popularity()
    return item_popularity


def get_val_data(env):
    df_train, df_user, df_item, list_feat = None, None, None, None
    if env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        df_val, df_user_val, df_item_val, list_feat = CoatEnv.get_df_coat("test.ascii")
    elif env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        df_val, df_user_val, df_item_val, list_feat = KuaiRandEnv.get_df_kuairand("test_processed.csv")
    elif env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        df_val, df_user_val, df_item_val, list_feat = KuaiEnv.get_df_kuairec("small_matrix_processed.csv")
    elif env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        df_val, df_user_val, df_item_val, list_feat = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-test.txt")
    elif env == "EtsyEnv-v0":
        from environments.Etsydata.etsy import EtsyEnv
        df_val, df_user_val, df_item_val, list_feat = EtsyEnv.get_df_etsy("df_test.csv")
    return df_val, df_user_val, df_item_val, list_feat


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
    
    elif env == "EtsyEnv-v0":
        parser.set_defaults(is_userinfo=False)
        parser.set_defaults(is_binarize=False)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[])
        parser.add_argument("--yfeat", type=str, default="rating")
        parser.add_argument("--rating_threshold", type=float, default=1)
        parser.add_argument('--leave_threshold', default=0.01, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        

    parser.add_argument('--force_length', type=int, default=10)
    parser.add_argument("--top_rate", type=float, default=0.8)

    args_new = parser.parse_known_args()[0]
    args.__dict__.update(args_new.__dict__)
    if env == "KuaiEnv-v0":
        args.use_userEmbedding = False

    return args


def get_true_env(args, read_user_num=None):
    if args.env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        mat, df_item, mat_distance = CoatEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "df_item": df_item,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}
        env = CoatEnv(**kwargs_um)
        env_task_class = CoatEnv
    elif args.env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        mat, mat_distance = YahooEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}

        env = YahooEnv(**kwargs_um)
        env_task_class = YahooEnv
    elif args.env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        mat, list_feat, mat_distance = KuaiRandEnv.load_mat(args.yfeat, read_user_num=read_user_num)
        kwargs_um = {"yname": args.yfeat,
                     "mat": mat,
                     "mat_distance": mat_distance,
                     "list_feat": list_feat,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}
        env = KuaiRandEnv(**kwargs_um)
        env_task_class = KuaiRandEnv
    elif args.env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        mat, lbe_user, lbe_item, list_feat, df_video_env, df_dist_small = KuaiEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "lbe_user": lbe_user,
                     "lbe_item": lbe_item,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init,
                     "list_feat": list_feat,
                     "df_video_env": df_video_env,
                     "df_dist_small": df_dist_small}
        env = KuaiEnv(**kwargs_um)
        env_task_class = KuaiEnv
    elif args.env == "EtsyEnv-v0":
        from environments.Etsydata.etsy import EtsyEnv
        mat, df_item, mat_distance = EtsyEnv.load_mat()
        kwargs_um = {"mat": mat,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}
        env = EtsyEnv(**kwargs_um)
        env_task_class = EtsyEnv
    return env, env_task_class, kwargs_um