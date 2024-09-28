import sys
import torch
import argparse
from src.components.data_loader import HardDataset  # Assuming `data_loader` is under `src.components`
from src.components.data_processor import DataProcessor
from src.components.model_loader import ModelLoader
from src.components.model_trainer import ModelTrainer
from src.utils import set_seed  # Utility function for seed setting
from src.exception import CustomException

def main(args):
    try:
        # Set the seed for reproducibility
        set_seed(args.seed, torch.cuda.is_available())

        # Set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        trainer = ModelTrainer(model_loader=model_loader, train_dataset=train_dataset, val_dataset=val_dataset, tokenizer=tokenizer)

        # Start the training process
        trainer.train()

        # Evaluate the model on the test dataset
        evaluation_results = trainer.evaluate(test_dataset)
        print("Evaluation Results:", evaluation_results)
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

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)
