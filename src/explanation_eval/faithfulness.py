import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
        saliency_data = self.load_saliency_scores(self.saliency_scores_path)
        thresholds = list(range(0, 110, 10))
        model_scores = []
        y_pred = []

        for threshold in thresholds:
            updated_y_pred = []
            for entry in tqdm(saliency_data, desc=f'Computing faithfulness for threshold {threshold}%'):
                saliencies = entry['saliences']
                token_ids = self.dataset[entry['index']]["input_ids"]
                new_token_ids = self.perturb_salient_tokens(token_ids, saliencies, threshold, mask=True)
                updated_y_pred_label = self.predict(new_token_ids)
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

    def load_saliency_scores(self, path):
        with open(path, 'r') as file:
            data = [json.loads(line) for line in file]
        return data

    def perturb_salient_tokens(self, token_ids, saliencies, threshold, tokenizer, mask=True):
        """
        Masks the most salient tokens in a sequence based on given saliency scores and a threshold.
        """
        if len(token_ids) != len(saliencies):
            raise ValueError("Length of token_ids and saliencies must match.")

        n_tokens = len([_t for _t in token_ids if _t != tokenizer.pad_token_id])
        k = int((threshold / 100) * n_tokens)

        sorted_idx = np.array(saliencies).argsort()[::-1]
        new_token_ids = token_ids[:]

        if mask and k > 0:
            num_masked = 0
            for _id in sorted_idx:
                if _id < n_tokens and token_ids[_id] != tokenizer.pad_token_id:
                    new_token_ids[_id] = tokenizer.mask_token_id
                    num_masked += 1
                    if num_masked == k:
                        break

        return new_token_ids

    def predict(self, input_ids, attention_mask=None, token_type_ids=None):
        input_ids = torch.tensor([input_ids]).to(self.device) 
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        probas = F.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probas, dim=-1).cpu().item()
        return pred