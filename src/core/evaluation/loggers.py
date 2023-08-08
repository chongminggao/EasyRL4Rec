from logzero import logger
import re


class LoggerEval_UserModel():
    def __init__(self):
        super().__init__()
    
    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, logs))

        # 2. upload logger
        # self.upload_logger()


class LoggerEval_Policy():
    def __init__(self, force_length, metrics):
        self.force_length = force_length
        self.metrics = metrics  # 需和evaluator函数对应
    
    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):
        def find_item_domination_results(prefix):
            pattern = re.compile(prefix + "ifeat_")
            res_domination = {}
            for k,v in results.items():
                res_search = re.match(pattern, k)
                # print(k, res_search)
                if res_search:
                    res_domination[k] = v
            return res_domination

        def get_one_result(prefix):
            num_test = results["n/ep"]
            len_tra = results[prefix + "n/st"] / num_test
            R_tra = results[prefix + "rew"]
            ctr = R_tra / len_tra

            res = dict()
            res['num_test'] = num_test
            for metric in self.metrics:
                if metric == 'len_tra':
                    res[prefix + 'len_tra'] = len_tra
                elif metric == 'R_tra':
                    res[prefix + 'R_tra'] = R_tra
                elif metric == 'ctr':
                    res[prefix + 'ctr'] = f"{ctr:.5f}"
                elif metric in ['CV', 'CV_turn', 'Diversity', 'Novelty', 'Serendipity']:  # have been calculated by evaluators
                    res[prefix + metric] = f"{results[prefix + metric]:.5f}"
                
                elif metric == "all_feats" and (prefix + "all_feats") in results:
                    res[prefix + 'all_feats'] = results[prefix + "all_feats"].tolist()

            return res

        results_all = {}
        for prefix in ["", "NX_0_", f"NX_{self.force_length}_"]:
            res = get_one_result(prefix)
            results_all.update(res)

            if 'ifeat_' in self.metrics:
                res_domination = find_item_domination_results(prefix)
                results_all.update(res_domination)

        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, results_all))

        # 2. upload logger
        # self.upload_logger()