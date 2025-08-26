import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="heart.csv"):
    df = pd.read_csv(path)
    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["target"], random_state=42)
    symptom_cols = [c for c in df.columns if c != "target"]
    print(f"Shape train: {train_df.shape} Shape test: {test_df.shape}")
    return df, train_df, test_df, symptom_cols
