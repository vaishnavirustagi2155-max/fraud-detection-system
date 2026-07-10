import pandas as pd

def preprocess_input(df, model_features):
    df = df.copy()

    # Example column mapping
    column_mapping = {
        "amt": "Amount",
        "transaction_time": "Time"
    }

    df.rename(columns=column_mapping, inplace=True)

    # Align columns with training features
    df = df.reindex(columns=model_features, fill_value=0)

    return df