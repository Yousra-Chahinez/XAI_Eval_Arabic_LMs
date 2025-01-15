class RationaleProcessor:
    def __init__(self, model_name, csv_path, task_type, max_length=128):
        self.model_name = model_name
        self.csv_path = csv_path  
        self.task_type = task_type  
        self.max_length = max_length 
        self.df_annotations = None  
        self.preprocessor = ArabertPreprocessor(model_name=model_name)
        self.tokenizer = None  

    def load_data(self):
        self.df_annotations = pd.read_csv(self.csv_path, delimiter=";", encoding='utf-8')
        print("Data loaded successfully.")
        print(self.df_annotations.head(5))

    def load_model(self):
        """Load the model and tokenizer using the ModelLoader component."""
        model_loader = ModelLoader(model_name=self.model_name)
        model_loader.setup()  # Load tokenizer, config, and model
        self.tokenizer = model_loader.tokenizer
        print("Model and tokenizer loaded successfully.")

    @staticmethod
    def split_arabic_text(text):
        """Splits Arabic text into words."""
        return text.split()

    def preprocess_arabic_text(self, text):
        """Preprocess Arabic text using AraBERT preprocessor."""
        return self.preprocessor.preprocess(text)

    def majority_vote(self, rationales):
        """Aggregates rationales and returns majority voted words."""
        # Create a set for each rationale to eliminate duplicates within the same rationale
        unique_rationale_words = [set(rationale) for rationale in rationales]
         # Flatten the list of sets to a single list while counting occurrences
        all_words = [word for rationale in unique_rationale_words for word in rationale]
        word_counts = Counter(all_words)
        # Filter words that appear in at least two of the three annotations
        majority_words = [word for word, count in word_counts.items() if count >= 2]  # Majority rule
        return majority_words
    
    def create_binary_scores(self, text_input, majority_vote):
        """Creates binary scores after tokenizing input text and human majority-voted text."""
        if self.task_type == "ASA":
            full_inputs = self.tokenizer(
                text_input,
                max_length=self.max_length, truncation=True,
                add_special_tokens=True, return_attention_mask=True)
        elif self.task_type == "Q2Q":
            full_inputs = self.tokenizer(
                text_input[0], text_input[1],
                max_length=self.max_length, truncation=True,
                add_special_tokens=True, return_attention_mask=True)

        full_input_ids = full_inputs['input_ids']
        # Tokenize the majority-voted text
        majority_txt = ' '.join(majority_vote)
        majority_words_token_ids = self.tokenizer.encode(majority_txt, add_special_tokens=False)

        # Create binary scores by comparing token IDs
        binary_scores = [1 if token in majority_words_token_ids else 0 for token in full_input_ids]
        return full_input_ids, binary_scores

    def construct_rationales(self):
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
            self.df_annotations[['input_ids', 'human_rationales']] = self.df_annotations.apply(
                lambda row: pd.Series(self.create_binary_scores(row['preprocessed_text'], row['majority_vote'])), axis=1
            )
            print("Human rationales constructed successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def compute_AP(self, row, saliency_col):
        text_input = (row['question1'], row['question2']) if self.task_type == "Q2Q" else row['text']

        # Retrieve token IDs and binary scores based on majority-voted words
        full_input_ids, human_rationales = self.create_binary_scores(text_input, row['majority_vote'])

        # Check if the majority vote is empty
        if not row['majority_vote']:
            return 0.0

        # Retrieve model saliency scores
        saliences = row[saliency_col]
        if isinstance(saliences, str):
            saliences = ast.literal_eval(saliences)

        # Filter out [CLS], [SEP], [PAD] tokens
        input_tokens = self.tokenizer.convert_ids_to_tokens(full_input_ids)
        content_indices = [i for i, token in enumerate(input_tokens) if token not in ['[CLS]', '[SEP]', '[PAD]']]
        filtered_human_rationales = [human_rationales[i] for i in content_indices]
        filtered_sal_scores = [saliences[i] for i in content_indices]

        if sum(filtered_human_rationales) == 0 or len(filtered_human_rationales) != len(filtered_sal_scores):
            return 0.0
        
        # Compute AP between human scores and model scores
        ap_score = average_precision_score(filtered_human_rationales, filtered_sal_scores)
        return ap_score

    def evaluate_plausibility(self, saliency_col):
        """Computes the Average Precision (AP) for each row and returns the Mean Average Precision (MAP)."""
        # Compute AP scores for each row in the DataFrame.
        self.df_annotations[f'ap_{saliency_col}'] = self.df_annotations.apply(
            lambda row: self.compute_AP(row, saliency_col), axis=1
        )

        # Calculate the Mean Average Precision (MAP)
        map_value = self.df_annotations[f'ap_{saliency_col}'].dropna().mean()
        print(f"Mean Average Precision (MAP) for {saliency_col}: {map_value:.4f}")
        return self.df_annotations, map_value
    
    def save_to_csv(self, output_path):
        """Saves the final processed DataFrame to a CSV file."""
        self.df_annotations = self.df_annotations.drop('preprocessed_text', axis=1)
        self.df_annotations.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
