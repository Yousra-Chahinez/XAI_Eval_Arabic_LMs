import sys
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from captum.attr import ShapleyValueSampling
from src.exception import CustomException
from src.logger import logging
from src.components.model_loader import ModelLoader

class LIMEExplainer:
    """
    Class to generate LIME explanations for a model's predictions.
    """
    def __init__(self, model_loader: ModelLoader, device='cpu'):
        """
        Initialize the LIMEExplainer with a loaded model and tokenizer.

        Parameters:
            model_loader (ModelLoader): Instance of the ModelLoader class with loaded model, tokenizer, and config.
            device (str): Device to use for computation ('cpu' or 'cuda').
        """
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = device

    def predictor(self, token_ids_list):
        """
        Predictor function required by LIME for generating predictions.
        
        Parameters:
            token_ids_list (list): List of tokenized input strings.

        Returns:
            np.ndarray: Array of logits for each input instance.
        """
        try:
            self.model.eval()
            all_logits = []
            for token_ids_str in token_ids_list:
                # Convert token string to list of integers
                token_ids = [int(token_id) for token_id in token_ids_str.split()]
                if not token_ids:
                    token_ids = [self.tokenizer.pad_token_id]

                # Ensure the tensor is of the correct type (long integers)
                input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids.long(), attention_mask=input_ids.long() > 0)
                    logits = outputs.logits[0].detach().cpu().numpy().tolist()
                    all_logits.append(logits)

            return np.array(all_logits)
        except Exception as e:
            logging.error("Error in LIME predictor function")
            raise CustomException(e, sys)

    def explain_prediction(self, text, task, max_length=128):
        """
        Generate LIME explanations for a given text based on the model's predictions.

        Parameters:
            text (str): Input text or a pair of texts for the task.
            task (str): Type of task ('ASA' for Arabic Sentiment Analysis, 'Q2Q' for question-to-question).
            max_length (int): Maximum length of tokens for input.

        Returns:
            list: Saliency values corresponding to each token.
        """
        try:
            # Encode the text based on the task type
            if task == 'ASA':
                token_ids = self.tokenizer.encode(text, max_length=max_length, truncation=True)
            else:  # Q2Q Task
                token_ids = self.tokenizer.encode(text[0], text[1], max_length=45, truncation=True)

            # Handle short sequences by padding
            if len(token_ids) < 6:
                token_ids = token_ids + [self.tokenizer.pad_token_id] * (6 - len(token_ids))

            # Prepare input for LIME
            token_ids_str = " ".join([str(i) for i in token_ids])
            explainer = LimeTextExplainer()
            exp = explainer.explain_instance(token_ids_str, self.predictor, num_features=len(token_ids), num_samples=1000)
            
            top_label = exp.available_labels()[0]
            explanation = exp.as_list(label=top_label)
            token_explanation_map = {int(w): s for w, s in explanation}

            # Create saliencies list
            saliencies = [token_explanation_map.get(token_id, None) for token_id in token_ids]
            return saliencies
        except Exception as e:
            logging.error("Error in explain_prediction method for LIME")
            raise CustomException(e, sys)

class SHAPExplainer:
    """
    Class to generate SHAP explanations for a model's predictions.
    """
    def __init__(self, model_loader: ModelLoader, device='cpu'):
        """
        Initialize the SHAPExplainer with a loaded model and tokenizer.

        Parameters:
            model_loader (ModelLoader): Instance of the ModelLoader class with loaded model, tokenizer, and config.
            device (str): Device to use for computation ('cpu' or 'cuda').
        """
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = device

    def predictor_shap(self, input_ids, attention_mask):
        """
        Predictor function required by SHAP for generating predictions.

        Parameters:
            input_ids (torch.Tensor): Tensor of input IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.

        Returns:
            torch.Tensor: Logits for each input instance.
        """
        try:
            self.model.eval()
            inputs = {'input_ids': input_ids.to(self.device), 'attention_mask': attention_mask.to(self.device)}
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.logits  # Directly return logits as tensor
        except Exception as e:
            logging.error("Error in SHAP predictor function")
            raise CustomException(e, sys)

    def explain_prediction(self, input_ids, attention_mask, target_class_idx):
        """
        Generate SHAP explanations for a given input based on the model's predictions.

        Parameters:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            attention_mask (torch.Tensor): Attention mask for input.
            target_class_idx (int): Target class index for explanation.

        Returns:
            list: SHAP scores corresponding to each token.
        """
        try:
            # Initialize the Shapley Value Sampling explainer
            ablator = ShapleyValueSampling(self.predictor_shap)
            
            # Compute attributions for the specified class
            attributions = ablator.attribute(input_ids, target=target_class_idx, additional_forward_args=(attention_mask,))
            attributions = attributions.detach().cpu().numpy()
            return attributions[0].tolist()
        except Exception as e:
            logging.error(f"Error in explain_prediction method for SHAP with target class {target_class_idx}")
            raise CustomException(e, sys)
