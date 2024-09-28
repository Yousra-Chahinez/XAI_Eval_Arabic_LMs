import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

class ModelLoader:
    def __init__(self, model_name, num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.config = None
        self.model = None
        
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Tokenizer loaded for {self.model_name}")

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self.model_name, 
                                                 num_labels=self.num_labels,
                                                 hidden_dropout_prob=0.1,
                                                 attention_probs_dropout_prob=0.1)
        print(f"Config loaded for {self.model_name}")

    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        print(f"Model loaded for {self.model_name}")
    
    def setup(self):
        self.load_tokenizer()
        self.load_config()
        self.load_model()