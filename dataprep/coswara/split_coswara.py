import pandas as pd
import json
import numpy as np

from sklearn.model_selection import train_test_split

from pathlib import Path

COSWARA_METADATA_PATH = 'data/iiscleap-Coswara-Data-bf300ae/combined_data.csv'
COSWARA_DURS_PATH = 'data/coswara_durations.csv'
COSWARA_BASE_PATH = 'data/iiscleap-Coswara-Data-bf300ae/Extracted_data'
AUDIO_COL = 'cough-heavy'
AGE_COL = 'a'
GENDER_COL = 'g'
SEED = 0
COSWARA_TRAIN_PATH = 'data/coswara_train.json'
COSWARA_EVAL_PATH = 'data/coswara_eval.json'
COSWARA_TRAIN_PATH_NO_BALANCE = 'data/coswara_train_no_balance.json'
COSWARA_EVAL_PATH_NO_BALANCE = 'data/coswara_eval_no_balance.json'

coswara_map = {
    'healthy': 'COVID_negative',
    'no_resp_illness_exposed': 'exclude',
    'positive_asymp': 'COVID_positive',
    'positive_mild': 'COVID_positive',
    'positive_moderate': 'COVID_positive',
    'recovered_full': 'exclude',
    'resp_illness_not_identified': 'exclude',
    'under_validation': 'exclude',
}

def read_coswara_metadata():
    df = pd.read_csv(COSWARA_METADATA_PATH)
    df['label'] = df['covid_status'].map(coswara_map)
    
    return df

def read_coswara_durations():
    df = pd.read_csv(COSWARA_DURS_PATH)
    return df

def merge_coswara_metadata_and_durations(metadata: pd.DataFrame, durations: pd.DataFrame):
    return metadata.merge(durations, on='id', how='left', validate='1:1')

def filter_coswara(metadata_with_durations: pd.DataFrame):
    df_filtered = metadata_with_durations.loc[metadata_with_durations[AUDIO_COL] > 0]
    df_filtered = df_filtered[df_filtered['label'] != 'exclude']
    df_filtered = df_filtered.dropna(subset=[AGE_COL, GENDER_COL])

    print(f"Filtered {len(metadata_with_durations)} rows to {len(df_filtered)} rows")

    return df_filtered


def add_strata_cols(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 18, 30, 40, 50, 60, np.inf]
    labels = ['0-17', '18-29', '30-39', '40-49', '50-59', '60+']
    
    # Use .assign() to create new columns without SettingWithCopyWarning
    df_with_strata = df.assign(
        age_bin = pd.cut(df[AGE_COL], bins=bins, labels=labels, right=False)
    )
    
    df_with_strata = df_with_strata.assign(
        strata = (
            df_with_strata['age_bin'].astype(str) + '_' + 
            df_with_strata[GENDER_COL].fillna('Unknown').astype(str)
        )
    )
    return df_with_strata

def balance_coswara(metadata_clean: pd.DataFrame):
    """
    Balances the dataset by matching the majority class's age/gender
    distribution to the minority class's distribution.
    """    
    positive_samples = metadata_clean[metadata_clean['label'] == 'COVID_positive']
    negative_samples = metadata_clean[metadata_clean['label'] == 'COVID_negative']

    if len(positive_samples) > len(negative_samples):
        df_majority = positive_samples
        df_minority = negative_samples
    else:
        df_majority = negative_samples
        df_minority = positive_samples
        
    print(f"Minority class size: {len(df_minority)} samples.")
    print(f"Majority class size: {len(df_majority)} samples.")

    df_minority = add_strata_cols(df_minority)
    df_majority = add_strata_cols(df_majority)
    
    target_strata_counts = df_minority['strata'].value_counts()
    print("\nTarget distribution from minority class:")
    print(target_strata_counts)

    # 5. (User Step 2) Sample similar in the major class
    print("\nSampling majority class to match target distribution...")
    sampled_groups = []
    
    for strata_name, target_count in target_strata_counts.items():
        # Find all matching samples in the majority class
        majority_strata_group = df_majority[df_majority['strata'] == strata_name]
        current_available = len(majority_strata_group)

        if current_available == 0:
            print(f"  WARNING: No samples in majority class for stratum '{strata_name}'. Skipping.")
            continue
        
        if current_available < target_count:
            print(f"  WARNING: Stratum '{strata_name}': "
                  f"target is {target_count}, but only {current_available} available. Using all.")
            sampled_groups.append(majority_strata_group)
        else:
            sampled_groups.append(
                majority_strata_group.sample(target_count, random_state=SEED)
            )

    if not sampled_groups:
        print("ERROR: No samples were selected. Check your data and strata.")
        return pd.DataFrame() # Return empty df
        
    df_majority_balanced = pd.concat(sampled_groups)
    
    df_balanced = pd.concat([df_majority_balanced, df_minority])
    
    print(f"\nFinal balanced dataset size: {len(df_balanced)} rows.")
    
    return df_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)

def split_coswara(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df['label'])

def create_coswara_filelist(coswara_df: pd.DataFrame) -> dict:
    coswara_filelist = {'data': []}
    for i, row in coswara_df.iterrows():
        audio_path = f'{COSWARA_BASE_PATH}/{row["id"]}/{AUDIO_COL}.wav'
        if not Path(audio_path).exists():
            print(f'{audio_path} does not exist')
            continue

        covid_status = row['label']
        
        coswara_filelist['data'].append(
            {'wav': audio_path,
            'labels': covid_status}
        )
    return coswara_filelist

if __name__ == '__main__':
    coswara_df = read_coswara_metadata()
    durations = read_coswara_durations()
    coswara_df = merge_coswara_metadata_and_durations(coswara_df, durations)
    coswara_df = filter_coswara(coswara_df)
    balanced_df = balance_coswara(coswara_df)
    train_df, eval_df = split_coswara(balanced_df)
    train_filelist = create_coswara_filelist(train_df)
    eval_filelist = create_coswara_filelist(eval_df)
    with open(COSWARA_TRAIN_PATH, 'w') as f:
        json.dump(train_filelist, f, indent=4)
    with open(COSWARA_EVAL_PATH, 'w') as f:
        json.dump(eval_filelist, f, indent=4)
