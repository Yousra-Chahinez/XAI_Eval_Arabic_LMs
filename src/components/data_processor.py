# import arabert
import sys
from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
from datasets import Dataset 
from sklearn.model_selection import train_test_split
from src.exception import CustomException

SEED = 42

class DataProcessor:
    def __init__(self, df, model_name, seed=SEED, task_type="ASA"):
        self.df = df
        self.model_name = model_name
        self.seed = SEED
        self.task_type = task_type
        self.preprocessor = ArabertPreprocessor(model_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_text(self, text):
        return self.preprocessor.preprocess(text)    
    
    def preprocess_data(self):
        try:
            """
            Apply preprocessing to the entire dataset based on the task type.
            """
            if self.task_type == "ASA":
                self.df['text'] = self.df['text'].apply(self.preprocess_text)
            elif self.task_type == "Q2Q":
                self.df['question1'] = self.df['question1'].apply(self.preprocess_text)
                self.df['question2'] = self.df['question2'].apply(self.preprocess_text)
            print(f"Preprocessing completed for {self.task_type} task using {self.model_name}.")
        except Exception as e:
            raise CustomException(e, sys)
        
    def split_data(self, test_size=0.1, val_size=0.1111):
        try:
            # Split into train, validation, and test sets
            train_val_df, test_df = train_test_split(self.df, test_size=test_size, random_state=self.seed, stratify=self.df['label'])
            train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=self.seed, stratify=train_val_df['label'])

            # Convert the DataFrames to Hugging Face Dataset objects
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
            test_dataset = Dataset.from_pandas(test_df)

            print(f"Data splitting completed. Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
        except Exception as e:
            raise CustomException(e, sys)
        return train_dataset, val_dataset, test_dataset

    def tokenize_data(self, df, max_length=128):
        try:
            print("Starting tokenization...")
            if self.task_type == "ASA":
                tokenized_data = self.tokenizer(
                    list(df['text']),
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
            elif self.task_type == "Q2Q":
                tokenized_data = self.tokenizer(
                    list(df['question1']),
                    list(df['question2']),
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    truncation=True,
                    padding='max_length',
                    max_length=45,  # Specific to Q2Q
                    return_attention_mask=True,
                    return_tensors="pt"
                )
            print(f"Tokenization completed for {self.task_type} task.")
        except Exception as e:
            raise CustomException(e, sys)
        return tokenized_data
