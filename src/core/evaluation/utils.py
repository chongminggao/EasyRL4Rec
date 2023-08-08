import numpy as np

def get_feat_dominate_dict(df_item_val, all_acts_origin, item_feat_domination, top_rate=0.8, draw_bar=False):
    if item_feat_domination is None:  # for yahoo
        return dict()
    # if need_transform:
    #     all_acts_origin = lbe_item.inverse_transform(all_acts)
    # else:
    #     all_acts_origin = all_acts

    feat_dominate_dict = {}
    recommended_item_features = df_item_val.loc[all_acts_origin]

    if "feat" in item_feat_domination:  # for kuairec and kuairand
        sorted_items = item_feat_domination["feat"]
        values = np.array([pair[1] for pair in sorted_items])
        values = values / sum(values)
        cumsum = values.cumsum()
        ind = 0
        for v in cumsum:
            if v > top_rate:
                break
            ind += 1
        if ind == 0:
            ind += 1
        dominated_values = np.array([pair[0] for pair in sorted_items])
        dominated_values = dominated_values[:ind]

        # dominated_value = sorted_items[0][0]
        recommended_item_features = recommended_item_features.filter(regex="^feat", axis=1)
        feat_numpy = recommended_item_features.to_numpy().astype(int)

        dominate_array = np.zeros([len(feat_numpy)], dtype=bool)
        for value in dominated_values:
            equal_mat = (feat_numpy == value)
            has_dominate = equal_mat.sum(axis=1)
            dominate_array = dominate_array | has_dominate

        rate = dominate_array.sum() / len(recommended_item_features)
        feat_dominate_dict["ifeat_feat"] = rate

        ####################################
        #Todo: visual bar plot
        if draw_bar:
            feat_predicted = feat_numpy
            cats_predicted = feat_predicted.reshape(-1)
            pos_cat_predicted = cats_predicted[cats_predicted > 0]

            feat_dominate_dict["all_feats"] = pos_cat_predicted

        ####################################

    else:  # for coat
        for feat_name, sorted_items in item_feat_domination.items():
            values = np.array([pair[1] for pair in sorted_items])

            values = values / sum(values)
            cumsum = values.cumsum()
            ind = 0
            for v in cumsum:
                if v > top_rate:
                    break
                ind += 1
            if ind == 0:
                ind += 1
            dominated_values = np.array([pair[0] for pair in sorted_items])
            dominated_values = dominated_values[:ind]

            # recommended_item_features = recommended_item_features.filter(regex="^feat", axis=1)
            feat_numpy = recommended_item_features[feat_name].to_numpy().astype(int)

            dominate_array = np.zeros([len(feat_numpy)], dtype=bool)
            for value in dominated_values:
                has_dominate = (feat_numpy == value)
                # has_dominate = equal_mat
                dominate_array = dominate_array | has_dominate

            rate = dominate_array.sum() / len(recommended_item_features)

            # dominated_value = sorted_items[0][0]
            # rate = (recommended_item_features[feat_name] == dominated_value).sum() / len(recommended_item_features)
            feat_dominate_dict["ifeat_" + feat_name] = rate

    return feat_dominate_dict

