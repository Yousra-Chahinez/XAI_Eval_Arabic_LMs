import sys
import torch
import argparse
from src.components.data_loader import HardDataset  # Assuming `data_loader` is under `src.components`
from src.components.data_processor import DataProcessor
from src.components.model_loader import ModelLoader
from src.components.model_trainer import ModelTrainer
from src.utils import set_seed  # Utility function for seed setting
from src.exception import CustomException
from src.utils import Visualizer
from src.explanation_methods.gradients_methods import vanilla_grad, gradient_input, IntegratedGradients, get_gradients  
from src.explanation_methods.perturbation_methods import LIMEExplainer, SHAPExplainer


# Compute explanation for a single instance using each XAI method
def run_explanation(args, model_loader, test_dataset):
    """
    Function to run explanation methods and visualize the results.
    """
    try:
        # Choose a specific sample from the dataset
        selected_instance = test_dataset.select([args.instance_index])[0]
        input_ids = torch.tensor([selected_instance['input_ids']])
        attention_mask = torch.tensor([selected_instance['attention_mask']])

        # Choose the target class index (assuming binary classification)
        target_class_idx = args.target_class

        # Get the explanation based on the chosen method
        if args.explanation_method == "vanilla_grad":
            gradients, _ = get_gradients(input_ids, attention_mask, target_class_idx, model_loader.model, device=args.device)
            attributions = vanilla_grad(gradients)
        elif args.explanation_method == "gradient_input":
            gradients, input_embeddings = get_gradients(input_ids, attention_mask, target_class_idx, model_loader.model, device=args.device)
            attributions = gradient_input(gradients, input_embeddings)
        elif args.explanation_method == "integrated_gradients":
            ig = IntegratedGradients(model=model_loader.model, device=args.device, n_steps=args.n_steps)
            print(input_ids[0])
            attributions = ig.compute_integrated_gradients(input_ids=input_ids[0], attention_mask=attention_mask[0], target_class_idx=target_class_idx)
        elif args.explanation_method == "lime":
            LIME_explainer = LIMEExplainer(model_loader, device=args.device)
            attributions = LIME_explainer.lime_explainer(selected_instance["text"], args.task_type)
            print(attributions)
        elif args.explanation_method == "shap_vs":
            SHAP_explainer = SHAPExplainer(model_loader, device=args.device)
            attributions = SHAP_explainer.shap_explainer(input_ids, attention_mask, target_class_idx)
        else:
            raise ValueError(f"Unsupported explanation method: {args.explanation_method}")

        # Visualize the attributions
        # Convert input_ids to tokens using the tokenizer
        tokenizer = model_loader.tokenizer
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        # Visualize the attributions using the Visualizer class
        visualizer = Visualizer(cmap='seismic')
        visualizer.visualize(tokens, attributions)
    except Exception as e:
        raise CustomException(e, sys)

def main(args):
    try:
        # Set the seed for reproducibility
        set_seed(args.seed, torch.cuda.is_available())

        # Set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.device = device 
        print("Device:", device)

        # Load the dataset
        dataset = HardDataset(args.dataset_path, args.seed)
        dataset.load_data()

        # Reduce the data size for computational efficiency (Optional)
        if args.reduce_data:
            reduced_df = dataset.reduce_data_size(size=args.reduce_size)
        else:
            reduced_df = dataset.df

        # Initialize the model loader and load the tokenizer, config, and model
        model_loader = ModelLoader(model_name=args.model_name)
        model_loader.setup()  # Ensure that tokenizer, config, and model are loaded

        # Initialize the data processor and preprocess the data
        data_processor = DataProcessor(reduced_df, model_name=model_loader.model_name, seed=args.seed, task_type=args.task_type)
        data_processor.preprocess_data()

        # Split the data into training, validation, and test sets
        train_dataset, val_dataset, test_dataset = data_processor.split_data(test_size=args.test_size, val_size=args.val_size)

        # Tokenize datasets
        train_dataset = train_dataset.map(data_processor.tokenize_data, batched=True)
        val_dataset = val_dataset.map(data_processor.tokenize_data, batched=True)
        test_dataset = test_dataset.map(data_processor.tokenize_data, batched=True)

        # Retrieve the tokenizer from ModelLoader
        tokenizer = model_loader.tokenizer

        # Initialize the trainer with the processed data and model
        # trainer = ModelTrainer(model_loader=model_loader, train_dataset=train_dataset, val_dataset=val_dataset, tokenizer=tokenizer)

        # Start the training process
        # trainer.train()

        # Evaluate the model on the test dataset
        # evaluation_results = trainer.evaluate(test_dataset)
        # print("Evaluation Results:", evaluation_results)

        # Evaluate the model on the test dataset
        # evaluation_results = trainer.evaluate(test_dataset)
        # print("Evaluation Results:", evaluation_results)

        # Run explanation method and visualize the results
        run_explanation(args, model_loader, test_dataset)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic NLP Model Training and Evaluation Pipeline")

    # Define command-line arguments for the script
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name or path (e.g., aubmindlab/bert-base-arabertv2)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file (e.g., data/HARD_balanced-reviews.tsv)')
    parser.add_argument('--task_type', type=str, default="ASA", choices=["ASA", "Q2Q"], help='Type of task (ASA or Q2Q)')
    parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of the data to be used as test set')
    parser.add_argument('--val_size', type=float, default=0.1111, help='Proportion of the training data to be used as validation set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--reduce_data', action='store_true', help='Flag to indicate if the data should be reduced for computational efficiency')
    parser.add_argument('--reduce_size', type=int, default=17000, help='Number of samples to reduce the dataset to (only used if --reduce_data is set)')
    parser.add_argument('--explanation_method', type=str, default="vanilla_grad", choices=["vanilla_grad", "gradient_input", "integrated_gradients", "lime", "shap_vs"], help='Explanation method to use')
    parser.add_argument('--instance_index', type=int, default=0, help='Index of the instance to explain from the test dataset')
    parser.add_argument('--target_class', type=int, default=1, help='Target class index for explanation')
    parser.add_argument('--n_steps', type=int, default=50, help='Number of steps for Integrated Gradients method')


    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)


