import numpy as np
import torch

from hierarchical_smoothing.cert import *
from hierarchical_smoothing.datasets import *
from hierarchical_smoothing.utils import *

from hierarchical_smoothing.graphs.models import *
from hierarchical_smoothing.graphs.training import *
from hierarchical_smoothing.graphs.prediction import *
from hierarchical_smoothing.graphs.smoothing import *

from hierarchical_smoothing.images.models import *
from hierarchical_smoothing.images.training import *
from hierarchical_smoothing.images.prediction import *
from hierarchical_smoothing.images.smoothing import *


class Experiment():

    def run(self, hparams):
        if hparams["datatype"] == "images":
            return self.run_image_experiment(hparams)
        elif hparams["datatype"] == "graphs":
            return self.run_graph_experiment(hparams)

    def run_image_experiment(self, hparams):
        results = {}
        dict_to_save = {}

        seed = int(torch.load(hparams["fixed_random_seeds_path"])[0])
        set_random_seed(seed)

        train_data, test_data_small, test_data = load_dataset(hparams,
                                                              seed=seed)
        model = create_image_classifier(hparams)
        model = train_image_classifier(model, train_data, hparams)

        model.eval()
        if not hparams['protected']:
            results["acc_short"] = predict_unprotected_images(hparams,
                                                              model,
                                                              test_data_small)
            results["acc"] = predict_unprotected_images(hparams,
                                                        model,
                                                        test_data)
            dict_to_save['model'] = model.state_dict()
            return results, dict_to_save

        pre_votes, targets = smooth_image_classifier(
            hparams, model, test_data_small, hparams["n0"])
        votes, _ = smooth_image_classifier(
            hparams, model, test_data_small, hparams["n1"])

        y_hat = pre_votes.argmax(1)
        y = torch.tensor(targets)
        correct = (y_hat == y).numpy()
        clean_acc = correct.mean()

        dict_to_save = certify(correct, votes, pre_votes, hparams)
        dict_to_save["clean_acc"] = clean_acc
        dict_to_save["correct"] = correct.tolist()

        return results, dict_to_save

    def run_graph_experiment(self, hparams):
        results = {}
        dict_to_save = {}

        seeds = torch.load(hparams["fixed_random_seeds_path"])
        seeds = seeds[:hparams["num_seeds"]]
        for seed in seeds:
            dict_to_save[seed] = {}

            data = load_dataset(hparams, seed=seed)
            [A, X, y, n, d, nc, train, valid, test,
                idx_train, idx_valid, idx_test] = data
            data_train = prepare_graph_data(train, hparams['device'])
            data_valid = prepare_graph_data(valid, hparams['device'])
            data_test = prepare_graph_data(test, hparams['device'])

            set_random_seed(seed)
            model = create_gnn(hparams)

            training_data = (data_train, data_valid, idx_train, idx_valid)
            model = train_gnn_inductive(model, training_data, hparams)
            model.eval()

            if not hparams['protected']:
                acc = predict_unprotected_graphs(model, data_test, idx_test)
                dict_to_save[seed]["acc"] = acc
                continue

            pre_votes = smooth_graph_classifier(
                hparams, model, data_test, hparams["n0"])
            votes = smooth_graph_classifier(
                hparams, model, data_test, hparams["n1"])

            pre_votes = pre_votes[idx_test]
            votes = votes[idx_test]
            y_hat = pre_votes.argmax(1)
            y = data_test.y.cpu()
            correct = (y_hat == y).numpy()
            clean_acc = correct.mean()

            dict_to_save[seed] = certify(correct, votes, pre_votes, hparams)
            dict_to_save[seed]["clean_acc"] = clean_acc
            dict_to_save[seed]["correct"] = correct.tolist()

        if not hparams['protected']:
            accs = [dict_to_save[k]["acc"] for k in seeds]
            dict_to_save["acc"] = np.mean(accs), np.std(accs)
            return results, dict_to_save

        # AVG
        clean_accs = [dict_to_save[k]['clean_acc'] for k in seeds]
        dict_to_save['clean_acc'] = np.mean(clean_accs), np.std(clean_accs)

        abstains = [dict_to_save[k]['abstain_binary'] for k in seeds]
        dict_to_save['abstain_binary'] = np.mean(abstains), np.std(abstains)

        abstains = [dict_to_save[k]['abstain_multiclass'] for k in seeds]
        averaged_result = np.mean(abstains), np.std(abstains)
        dict_to_save['abstain_multiclass'] = averaged_result

        smoothing_config = hparams['smoothing_config']
        smoothing_distribution = smoothing_config['smoothing_distribution']
        if smoothing_distribution in ["sparse", "hierarchical_sparse"]:
            dict_to_save['binary'] = {
                "ratios": avg_results(dict_to_save, "binary",
                                      "ratios", seeds),
                "cert_acc": avg_results(dict_to_save, "binary",
                                        "cert_acc", seeds),
            }
            dict_to_save['multiclass'] = {
                "ratios": avg_results(dict_to_save, "multiclass",
                                      "ratios", seeds),
                "cert_acc": avg_results(dict_to_save, "multiclass",
                                        "cert_acc", seeds),
            }

            # cleanup
            for seed in seeds:
                del dict_to_save[seed]['binary']
                del dict_to_save[seed]['multiclass']

        return results, dict_to_save
