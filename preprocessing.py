# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def handel_missing_values(df, strategy='auto'):
    log = {}
    df_clean = df.copy()
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count == 0:
            continue
        col_type = df_clean[col].dtype
        if strategy == 'auto':
            if col_type == 'object' or col_type.name == 'category':
                df_clean[col] = df_clean[col].fillna('Missing')
                log[col] = f"Filled {missing_count} with 'Missing'"
            else:
                median = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median)
                log[col] = f"Filled {missing_count} with median={median}"
        elif strategy == 'drop':
            df_clean.dropna(subset=[col], inplace=True)
            log[col] = f"Dropped rows with missing in {col}"
        elif strategy == 'custom':
            fill_val = df_clean[col].mode()[0] if col_type == 'object' else df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(fill_val)
            log[col] = f"Filled {missing_count} with {fill_val}"
    return df_clean, log

def handel_outliers_iqr(df):
    df_clean = df.copy()
    log = {}
    numeric_cols = df_clean.select_dtypes(include=['int64','float']).columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = df_clean.shape[0]
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        after = df_clean.shape[0]
        removed = before - after
        if removed > 0:
            log[col] = f"Removed {removed} outliers [{lower:.2f}, {upper:.2f}]"
    return df_clean, log

def preprocess_pipeline(df, strategy='auto'):
    df, miss_log = handel_missing_values(df, strategy)
    df, outlier_log = handel_outliers_iqr(df)
    df = df.loc[:, df.nunique() > 1]
    df = pd.get_dummies(df, drop_first=True)
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, miss_log, outlier_log

def generate_visuals(df_before, df_after, output_dir='visuals'):
    os.makedirs(output_dir, exist_ok=True)
    df_before.hist(figsize=(12, 8))
    plt.suptitle("Histogram Before")
    plt.savefig(f"{output_dir}/hist_before.png")
    plt.close()

    df_after.hist(figsize=(12, 8))
    plt.suptitle("Histogram After")
    plt.savefig(f"{output_dir}/hist_after.png")
    plt.close()

    plt.figure(figsize=(10,6))
    sns.heatmap(df_after.corr(), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/heatmap.png")
    plt.close()

def save_logs(miss_log, outlier_log, filename="logs.txt"):
    with open(filename, 'w') as f:
        f.write("Missing Value Log:\n")
        for k,v in miss_log.items():
            f.write(f"{k}: {v}\n")
        f.write("\nOutlier Log:\n")
        for k,v in outlier_log.items():
            f.write(f"{k}: {v}\n")
