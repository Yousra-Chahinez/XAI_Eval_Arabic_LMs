import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from explanation_eval.utils_eval_exp import load_saliency_scores, perturb_salient_tokens, predict


class FaithfulnessEvaluator:
    def __init__(self, model, dataset, saliency_scores_path, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.saliency_scores_path = saliency_scores_path
        self.device = device

    def evaluate_model(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def eval_faithfulness(self):
        self.model.eval()
        saliency_data = load_saliency_scores(self.saliency_scores_path)
        thresholds = list(range(0, 110, 10))
        model_scores = []
        y_pred = []

        for threshold in thresholds:
            updated_y_pred = []
            for entry in tqdm(saliency_data, desc=f'Computing faithfulness for threshold {threshold}%'):
                saliencies = entry['saliences']
                token_ids = self.dataset[entry['index']]["input_ids"]
                new_token_ids = perturb_salient_tokens(token_ids, saliencies, threshold, mask=True)
                updated_y_pred_label = predict(new_token_ids)
                updated_y_pred.append(updated_y_pred_label)

                if threshold == 0:
                    y_pred.append(entry['label'])

            accuracy = self.evaluate_model(y_pred, updated_y_pred)
            model_scores.append(accuracy)

        # Calculate the AUC and print it
        auc_score = auc(thresholds, model_scores)
        print(f"AUC Score: {auc_score:.2f}")

        # Prepare results data for saving
        results_data = {
            "thresholds": thresholds,
            "model_scores": model_scores,
            "AUC_score": auc_score
        }

        # Specify the path where you want to save the results
        results_path = '/data/results.json'

        # Save the results to a JSON file
        with open(results_path, 'w') as results_file:
            json.dump(results_data, results_file, indent=4)

        print(f"Results saved to {results_path}")
        return model_scores, auc_score
