import pandas as pd
from sklearn.model_selection import train_test_split
from config import SEED

def load_and_reduce_data(data_path, size=17000, seed=SEED):
    # Load the dataset
    df = pd.read_csv(data_path, delimiter='\t')
    
    # Keep only 'rating' and 'review' columns
    df = df[['rating', 'review']]
    
    # Convert rating: positive (1) if rating > 3, negative (0) if rating < 3
    df['rating'] = df['rating'].apply(lambda x: 0 if x < 3 else 1)
    
    # Rename columns for consistency with standard text classification format
    df.columns = ['label', 'text']
    print(f"Initial dataset length: {len(df)}")
    
    # Reduce the dataset size, stratified by label
    reduced_df, _ = train_test_split(df, train_size=size, stratify=df['label'], random_state=seed)
    
    print(f"Reduced dataset length: {len(reduced_df)}")
    print(reduced_df.head(5))
    
    return reduced_df
