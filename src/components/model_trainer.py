import sys
import time
import logging 
from transformers import TrainingArguments, Trainer
from src.utils import format_duration, compute_metrics
from src.components.data_loader import HardDataset
from src.components.data_processor import DataProcessor
from src.components.model_loader import ModelLoader
from src.exception import CustomException

class ModelTrainer():
    def __init__(self, model_loader, train_dataset, val_dataset, tokenizer):
        self.model_loader = model_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.trainer = None
    
    def model_init(self):
        # Ensure that the model is not None
        if self.model_loader.model is None:
            raise ValueError("Model is not initialized in ModelLoader.")
        return self.model_loader.model
    
    def train(self):
        # Define dynamic training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            save_strategy="no",      # Disable checkpoint saving
            save_total_limit=0,      # Set total limit of saved checkpoints to 0
            load_best_model_at_end=False,
        )

        # Initialize the Trainer with dynamic args
        self.trainer = Trainer(
            model_init=self.model_init,
            args=training_args,  # Dynamically set TrainingArguments
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        # Hyperparameter search
        start_time = time.time()
        try:
            best_run = self.trainer.hyperparameter_search(
                n_trials=10,
                direction="maximize",
                backend="optuna",
                hp_space=lambda trial: {
                    'learning_rate': trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                    'num_train_epochs': trial.suggest_int("num_train_epochs", 3, 5),
                    'per_device_train_batch_size': trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
                    'weight_decay': trial.suggest_float("weight_decay", 0.01, 0.1)
                }
            )
        except Exception as e:
            logging.error(f"Error during hyperparameter search: {e}")
            raise CustomException(e, sys)
        end_time = time.time()

        # Format and print the duration
        formatted_time = format_duration(end_time - start_time)
        print(f"Best hyperparameters found: {best_run}")
        logging.info(f"Best hyperparameters found: {best_run}")
        print(f"Training completed in {formatted_time}")
        logging.info(f"Training completed in {formatted_time}")

        # Set the trainer's args with the best hyperparameters
        for n, v in best_run.hyperparameters.items():
            setattr(self.trainer.args, n, v)

        # Start training with the best hyperparameters
        self.trainer.train()
    
    def evaluate(self, test_dataset):
        if self.trainer is None:
            raise ValueError("Trainer is not initialized. Run 'train()' first.")
        return self.trainer.evaluate(eval_dataset=test_dataset)
    
