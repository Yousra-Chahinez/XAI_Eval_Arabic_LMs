import sys
from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
from datasets import Dataset
from sklearn.model_selection import train_test_split
from utils import SEED
from src.exception import CustomException

def model_preprocessor(model_name, text):
    # Initialize the preprocessor and tokenizer
    preprocessor = ArabertPreprocessor(model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Preprocess the text
    preprocessed_text = preprocessor.preprocess(text)
    
    return preprocessed_text, tokenizer

def preprocess_data(df, preprocessor, task_type="ASA"):
    try:
        if task_type == "ASA":
            df['text'] = df['text'].apply(lambda x: model_preprocessor(preprocessor, x))
        elif task_type == "Q2Q":
            df['question1'] = df['question1'].apply(lambda x: model_preprocessor(preprocessor, x))
            df['question2'] = df['question2'].apply(lambda x: model_preprocessor(preprocessor, x))
        print(f"Preprocessing completed for {task_type} task using {preprocessor.model_name}.")
    except Exception as e:
        raise CustomException(e, sys)
    return df

def split_data(df, seed=SEED, test_size=0.1, val_size=0.1111):
    try:
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['label'])
        train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=seed, stratify=train_val_df['label'])

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        print(f"Data splitting completed. Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    except Exception as e:
        raise CustomException(e, sys)
    return train_dataset, val_dataset, test_dataset

def tokenize_data(df, tokenizer, task_type="ASA", max_length=128):
    try:
        print("Starting tokenization...")
        if task_type == "ASA":
            tokenized_data = tokenizer(
                list(df['text']),
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt"
            )
        elif task_type == "Q2Q":
            tokenized_data = tokenizer(
                list(df['question1']),
                list(df['question2']),
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=45,  
                return_attention_mask=True,
                return_tensors="pt"
            )
        print(f"Tokenization completed for {task_type} task.")
    except Exception as e:
        raise CustomException(e, sys)
    return tokenized_data
