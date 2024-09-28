import sys
import pandas as pd
from collections import Counter
from arabert.preprocess import ArabertPreprocessor
from src.components.model_loader import ModelLoader
from src.exception import CustomException
import argparse

# Construct human ratioanles from human annotations
class ConstructRationale:
    def __init__(self, model_name, csv_path):
        self.model_name = model_name
        self.csv_path = csv_path # annotation file
        self.df_annotations = None
        self.preprocessor = ArabertPreprocessor(model_name=model_name)

    def load_data(self):
        """Load the dataset from CSV."""
        self.df_annotations = pd.read_csv(self.csv_path, delimiter=";", encoding='utf-8')
        print(self.df_annotations.head(5))

    @staticmethod
    def split_arabic_text(text):
        """Splits Arabic text into words."""
        return text.split()

    def majority_vote(self, rationales):
        """Aggregates rationales and returns majority voted words."""
        # Create a set for each rationale to eliminate duplicates within the same rationale
        unique_rationale_words = [set(rationale) for rationale in rationales]

        # Flatten the list of sets to a single list while counting occurrences
        all_words = [word for rationale in unique_rationale_words for word in rationale]
        word_counts = Counter(all_words)

        # Filter words that appear in at least two of the three annotations
        majority_words = [word for word, count in word_counts.items() if count >= 2]
        return majority_words

    def preprocess_arabic_text(self, text):
        """Preprocess Arabic text using AraBERT preprocessor."""
        return self.preprocessor.preprocess(text)

    def create_binary_scores(self, row, tokenizer):
        """Creates binary scores after tokenizing input text and human majority-voted text."""
        # Tokenize the input text
        full_inputs = tokenizer(row['preprocessed_text'], max_length=128, truncation=True, add_special_tokens=True)
        full_input_ids = full_inputs['input_ids']

        # Tokenize the majority-voted text
        majority_words = row['majority_vote']
        majority_txt = ' '.join(majority_words)
        majority_words_token_ids = tokenizer.encode(majority_txt, add_special_tokens=False)

        # Create binary scores
        binary_scores = [1 if token in majority_words_token_ids else 0 for token in full_input_ids]
        return binary_scores

    def construct_ratioanles(self, tokenizer):
        try:
            """Processes annotations by splitting, preprocessing, and creating binary scores."""
            # Split the rationale columns into individual words
            self.df_annotations['annotation_1'] = self.df_annotations['annotation_1'].apply(self.split_arabic_text)
            self.df_annotations['annotation_2'] = self.df_annotations['annotation_2'].apply(self.split_arabic_text)
            self.df_annotations['annotation_3'] = self.df_annotations['annotation_3'].apply(self.split_arabic_text)

            # Preprocess the text column
            self.df_annotations['preprocessed_text'] = self.df_annotations['text'].apply(self.preprocess_arabic_text)

            # Aggregate rationales
            self.df_annotations['majority_vote'] = self.df_annotations[['annotation_1', 'annotation_2', 'annotation_3']].apply(self.majority_vote, axis=1)

            # Create binary scores
            self.df_annotations['human_rationales'] = self.df_annotations.apply(lambda row: self.create_binary_scores(row, tokenizer), axis=1)
        except Exception as e:
            raise CustomException(e, sys)

    def save_to_csv(self, output_path):
        """Saves the final processed dataframe to a CSV file."""
        self.df_annotations = self.df_annotations.drop('preprocessed_text', axis=1)
        self.df_annotations.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


def main(args):
    try:
        constructRationale = ConstructRationale(model_name=args.model_name, csv_path=args.csv_path)
        # Load data from the CSV
        constructRationale.load_data()

         # Load the model and tokenizer using ModelLoader
        model_loader = ModelLoader(model_name=args.model_name)
        model_loader.setup()  # Load tokenizer, config, and model

        # Construct the rationales
        constructRationale.construct_ratioanles(tokenizer=model_loader.tokenizer)
        # Save to the output file
        constructRationale.save_to_csv(args.output_path)

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Construct Human Rationales from Human Annotations")
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained Model Name')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing annotations')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV with rationales')

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)