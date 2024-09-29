import os
import sys
import torch
import argparse
import pickle  # New: For saving and loading preprocessed data
from src.components.data_loader import HardDataset
from src.components.data_processor import DataProcessor
from src.components.model_loader import ModelLoader
from src.components.model_trainer import ModelTrainer
from src.utils import set_seed
from src.exception import CustomException
from src.utils import Visualizer
from src.explanation_methods.gradients_methods import vanilla_grad, gradient_input, IntegratedGradients, get_gradients
from src.explanation_methods.perturbation_methods import LIMEExplainer, SHAPExplainer

# New: Function to save preprocessed datasets
def save_preprocessed_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Preprocessed data saved to {path}")

# New: Function to load preprocessed datasets
def load_preprocessed_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Preprocessed data loaded from {path}")
    return data

def run_explanation(args, model_loader, test_dataset):
    """
    Function to run explanation methods and visualize the results.
    """
    try:
        selected_instance = test_dataset.select([args.instance_index])[0]
        input_ids = torch.tensor([selected_instance['input_ids']])
        attention_mask = torch.tensor([selected_instance['attention_mask']])
        target_class_idx = args.target_class

        if args.explanation_method == "vanilla_grad":
            gradients, _ = get_gradients(input_ids, attention_mask, target_class_idx, model_loader.model, device=args.device)
            attributions = vanilla_grad(gradients)
            print(attributions)
        elif args.explanation_method == "gradient_input":
            gradients, input_embeddings = get_gradients(input_ids, attention_mask, target_class_idx, model_loader.model, device=args.device)
            attributions = gradient_input(gradients, input_embeddings)
        elif args.explanation_method == "integrated_gradients":
            ig = IntegratedGradients(model=model_loader.model, device=args.device, n_steps=args.n_steps)
            attributions = ig.compute_integrated_gradients(input_ids=input_ids[0], attention_mask=attention_mask[0], target_class_idx=target_class_idx)
        elif args.explanation_method == "lime":
            LIME_explainer = LIMEExplainer(model_loader, device=args.device)
            attributions = LIME_explainer.lime_explainer(selected_instance["text"], args.task_type)
        elif args.explanation_method == "shap_vs":
            SHAP_explainer = SHAPExplainer(model_loader, device=args.device)
            attributions = SHAP_explainer.shap_explainer(input_ids, attention_mask, target_class_idx)
        else:
            raise ValueError(f"Unsupported explanation method: {args.explanation_method}")

        tokenizer = model_loader.tokenizer
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        visualizer = Visualizer(cmap='seismic')
        visualizer.visualize(tokens, attributions)
    except Exception as e:
        raise CustomException(e, sys)

def train_and_evaluate(args):
    try:
        set_seed(args.seed, torch.cuda.is_available())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.device = device
        print("Device:", device)

        dataset = HardDataset(args.dataset_path, args.seed)
        dataset.load_data()

        reduced_df = dataset.reduce_data_size(size=args.reduce_size) if args.reduce_data else dataset.df
        model_loader = ModelLoader(model_name=args.model_name)
        model_loader.setup()

        data_processor = DataProcessor(reduced_df, model_name=model_loader.model_name, seed=args.seed, task_type=args.task_type)
        data_processor.preprocess_data()
        train_dataset, val_dataset, test_dataset = data_processor.split_data(test_size=args.test_size, val_size=args.val_size)
        train_dataset = train_dataset.map(data_processor.tokenize_data, batched=True)
        val_dataset = val_dataset.map(data_processor.tokenize_data, batched=True)
        test_dataset = test_dataset.map(data_processor.tokenize_data, batched=True)
        
        # Save the preprocessed datasets for later use
        preprocessed_data_path = args.preprocessed_data_path
        save_preprocessed_data((train_dataset, val_dataset, test_dataset), preprocessed_data_path)

        tokenizer = model_loader.tokenizer
        trainer = ModelTrainer(model_loader=model_loader, train_dataset=train_dataset, val_dataset=val_dataset, tokenizer=tokenizer)
        trainer.train()
        evaluation_results = trainer.evaluate(test_dataset)
        print("Evaluation Results:", evaluation_results)
    except Exception as e:
        raise CustomException(e, sys)

def main(args):
    if args.mode == 'train':
        train_and_evaluate(args)
    elif args.mode == 'explain':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.device = device  
        model_loader = ModelLoader(model_name=args.model_name)
        model_loader.setup()

        # Load the preprocessed data
        test_dataset_path = args.preprocessed_data_path
        _, _, test_dataset = load_preprocessed_data(test_dataset_path)

        run_explanation(args, model_loader, test_dataset)
    else:
        raise ValueError("Mode must be 'train' or 'explain'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic NLP Model Training, Evaluation, and Explanation Pipeline")

    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation: train or explain')

    # Training and Evaluation Arguments
    train_parser = subparsers.add_parser('train', help='Train and evaluate the model')
    train_parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name or path')
    train_parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    train_parser.add_argument('--task_type', type=str, default="ASA", choices=["ASA", "Q2Q"], help='Type of task (ASA or Q2Q)')
    train_parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of the data to be used as test set')
    train_parser.add_argument('--val_size', type=float, default=0.1111, help='Proportion of the training data to be used as validation set')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    train_parser.add_argument('--reduce_data', action='store_true', help='Flag to indicate if the data should be reduced for computational efficiency')
    train_parser.add_argument('--reduce_size', type=int, default=17000, help='Number of samples to reduce the dataset to (only used if --reduce_data is set)')
    train_parser.add_argument('--preprocessed_data_path', type=str, default='data/preprocessed_data.pkl', help='Path to save the preprocessed data for explanation')

    # Explanation Arguments
    explain_parser = subparsers.add_parser('explain', help='Generate explanations for the model predictions')
    explain_parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name or path')
    explain_parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    explain_parser.add_argument('--task_type', type=str, default="ASA", choices=["ASA", "Q2Q"], help='Type of task (ASA or Q2Q)')
    explain_parser.add_argument('--explanation_method', type=str, default="vanilla_grad", choices=["vanilla_grad", "gradient_input", "integrated_gradients", "lime", "shap_vs"], help='Explanation method to use')
    explain_parser.add_argument('--instance_index', type=int, default=0, help='Index of the instance to explain from the test dataset')
    explain_parser.add_argument('--target_class', type=int, default=1, help='Target class index for explanation')
    explain_parser.add_argument('--n_steps', type=int, default=50, help='Number of steps for Integrated Gradients method')
    explain_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    explain_parser.add_argument('--preprocessed_data_path', type=str, default='data/preprocessed_data.pkl', help='Path to load preprocessed data')

    args = parser.parse_args()
    main(args)
