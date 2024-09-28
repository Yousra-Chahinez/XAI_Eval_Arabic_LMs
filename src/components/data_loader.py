import pandas as pd
from sklearn.model_selection import train_test_split

class HardDataset:
    def __init__(self, data_path, seed):
        self.data_path=data_path
        self.seed=seed
        self.df=None
    
    def load_data(self,  size=17000):
        """
        Load the data from a TSV file and retain only the 'rating' and 'review' columns.
        Also, transform the 'rating' to a binary label (positive/negative).
        """
        # Load the dataset
        self.df = pd.read_csv(self.data_path, delimiter='\t')
        
        # Keep only 'rating' and 'review' columns
        self.df = self.df[['rating', 'review']]
        
        # Code rating: positive (1) if rating > 3, negative (0) if rating < 3
        self.df['rating'] = self.df['rating'].apply(lambda x: 0 if x < 3 else 1)
        
        # Rename columns for consistency with standard text classification format
        self.df.columns = ['label', 'text']
        print(f"Initial dataset length: {len(self.df)}")

        # Reduce the dataset size, stratified by label
        self.df, _ = train_test_split(self.df, train_size=size, 
                                         stratify=self.df['label'], random_state=self.seed)
        
        print(f"Reduced dataset length: {len(self.df)}")
        print(self.df.head(5))
        return self.df
        
