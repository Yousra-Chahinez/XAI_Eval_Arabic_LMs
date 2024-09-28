import sys
import numpy as np
import torch
from src.exception import CustomException
from src.logger import logging

class GradientComputations:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_gradients(self, input_ids, attention_mask, target_class_idx):
        try:
            self.model.to(self.device)
            self.model.eval()
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            embed_layer = self.model.bert.embeddings
            with torch.no_grad():
                input_embeds = embed_layer(input_ids)

            input_embeds.requires_grad_(True)
            self.model.zero_grad()

            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
            loss = outputs.logits[0, target_class_idx]
            loss.backward()

            if input_embeds.grad is not None:
                return input_embeds.grad.detach(), input_embeds
            else:
                raise CustomException("No gradients were computed.", sys)
        except Exception as e:
            logging.error(f"Error in get_gradients: {e}")
            raise CustomException(e, sys)

    def vanilla_grad(self, grads):
        try:
            grads = torch.tensor(grads)
            attributions = grads.abs().sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            return attributions.cpu().detach().numpy()
        except Exception as e:
            logging.error(f"Error in vanilla_grad: {e}")
            raise CustomException(e, sys)
    
    def gradient_input(self, grads, inputs_embed):
        try:
            x_grads = grads * inputs_embed
            x_grads = x_grads.sum(dim=-1).squeeze(0)
            x_grads = x_grads / torch.norm(x_grads)
            return x_grads.cpu().detach().numpy()
        except Exception as e:
            logging.error(f"Error in gradient_input: {e}")
            raise CustomException(e, sys)

class IntegratedGradients:
    def __init__(self, model, device, n_steps=50):
        self.model = model
        self.device = device
        self.n_steps = n_steps

    def interpolate_inputs(self, baseline, text):
        try:
            alphas = torch.linspace(0.0, 1.0, self.n_steps+1, device=self.device)
            alphas_x = alphas[:, None, None]
            text = text.to(self.device).float()
            baseline = baseline.to(self.device).float()
            delta = text - baseline
            return baseline + alphas_x * delta
        except Exception as e:
            logging.error("Error in interpolate_inputs method")
            raise CustomException(e, sys)

    def compute_integrated_gradients(self, input_ids, attention_mask, target_class_idx, baseline='zero'):
        try:
            embed_layer = self.model.bert.embeddings
            input_ids = input_ids.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)

            with torch.no_grad():
                sample_embed = embed_layer(input_ids)
            if baseline == 'zero':
                baseline_embed = torch.zeros_like(sample_embed)

            interpolated_inputs = self.interpolate_inputs(baseline_embed, sample_embed)
            attention_mask = attention_mask.repeat(interpolated_inputs.size(0), 1)

            gradients = GradientComputations(self.model, self.device)
            path_gradients, _ = gradients.get_gradients_embed(interpolated_inputs, attention_mask, target_class_idx)
            path_gradients = torch.tensor(path_gradients)
            all_grads = torch.sum(path_gradients, dim=0) / self.n_steps
            x_grads = all_grads * (sample_embed - baseline_embed)
            igs = torch.sum(x_grads, dim=-1).cpu().detach().numpy()

            norm = np.linalg.norm(igs, ord=2)
            igs = igs / norm
            return igs[0]
        except Exception as e:
            logging.error(f"Error in compute_integrated_gradients method for target class index {target_class_idx}")
            raise CustomException(e, sys)