import ast
import torch
import numpy as np
from sklearn.metrics import average_precision_score

class PlausibilityEvaluator:
    def __init__(self, df, tokenizer, max_length=128, task_type="ASA"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            tokenizer: The tokenizer used for tokenizing questions and rationales.
            max_length (int): The maximum length for tokenization.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_length = max_length

    def compute_ap(self, row, saliency_col):
        """
        Computes the Average Precision (AP) between human rationales and model-generated saliency scores.

        Args:
            row (pd.Series): A row from the DataFrame containing the question pairs and rationales.
            saliency_col (str): The column name where model saliency scores are stored.

        Returns:
            float: The computed AP score.
        """
        # Tokenize the combined questions (or text)

        if self.task_type == "ASA":
            full_inputs = self.tokenizer(
                row['text'], 
                max_length=self.max_length, truncation=True, 
                add_special_tokens=True, return_attention_mask=True)
        elif self.task_type == "Q2Q":
            full_inputs = self.tokenizer(
                row['question1'], row['question2'], 
                max_length=self.max_length, truncation=True, 
                add_special_tokens=True, return_attention_mask=True
            )
        full_input_ids = full_inputs['input_ids']

        if not row['majority_vote']:
            # Handle the case where majority vote is empty
            return 0.0

        # Process human rationales (majority votes)
        imp_words_human = ' '.join(row['majority_vote'])
        imp_token_ids_human = self.tokenizer.encode(imp_words_human, add_special_tokens=False)

        # Initialize saliency scores with zeros
        human_sal_scores = [0] * len(full_input_ids)

        # Assign score of 1 for tokens found in the human rationale
        for i, t in enumerate(full_input_ids):
            if t in imp_token_ids_human:
                human_sal_scores[i] = 1

        # Process model saliency scores from the specified column
        saliences = row[saliency_col]
        if isinstance(saliences, str):
            saliences = ast.literal_eval(saliences)

        input_tokens = self.tokenizer.convert_ids_to_tokens(full_input_ids)

        # Filter out non-content tokens (special tokens like [CLS], [SEP], [PAD])
        content_indices = [i for i, token in enumerate(input_tokens) if token not in ['[CLS]', '[SEP]', '[PAD]']]
        filtered_human_scores = [human_sal_scores[i] for i in content_indices]
        filtered_sal_scores = [saliences[i] for i in content_indices]

        # Check if there are positive human saliency scores
        if sum(filtered_human_scores) == 0:
            return 0.0

        # Compute Average Precision Score between filtered human scores and model saliency scores
        ap_score = average_precision_score(filtered_human_scores, filtered_sal_scores)
        return ap_score

    def evaluate_plausibility(self, saliency_col):
        """
        Applies the AP computation to each row in the DataFrame.

        Args:
            saliency_col (str): Column name where the model's saliency scores are stored.

        Returns:
            pd.DataFrame: DataFrame with an added column for AP scores.
        """
        self.df[f'ap_{saliency_col}'] = self.df.apply(
            lambda row: self.compute_ap(row, saliency_col), axis=1
        )
        return self.df

# Usage example
if __name__ == "__main__":
    # Assuming you have the DataFrame df_hm and a tokenizer initialized
    df_hm = ...  # Load your DataFrame
    tokenizer = ...  # Initialize your tokenizer
    
    plausibility_evaluator = PlausibilityEvaluator(df_hm, tokenizer, max_length=128)
    df_with_ap_scores = plausibility_evaluator.evaluate_plausibility('vg_saliences')

    # Now df_with_ap_scores contains the calculated AP scores
