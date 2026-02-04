import pandas as pd
import xarray as xr
import numpy as np

def get_lifetimes_table(datasets, dataset_labels=None):
    """
    Extracts decay lifetimes from pyglotaran result datasets and returns 
    a formatted table.

    Args:
        datasets (xr.Dataset or list): A single pyglotaran result dataset 
                                       or a list of datasets.
        dataset_labels (list, optional): A list of labels for the datasets. 
                                         If None, generic names are used.

    Returns:
        pd.DataFrame: A table containing the lifetimes for all provided samples.
    """
    # Standardize input to a list of datasets
    if not isinstance(datasets, list):
        datasets = [datasets]
    
    # Standardize labels
    if dataset_labels is None:
        dataset_labels = [f"Dataset_{i+1}" for i in range(len(datasets))]
    
    if len(datasets) != len(dataset_labels):
        raise ValueError("The number of datasets must match the number of labels.")

    lifetimes_dict = {}

    for ds, label in zip(datasets, dataset_labels):
        try:
            # Extract lifetimes: result.data[ds].lifetime_decay
            # .values is used to get the raw numpy array
            lts = ds.lifetime_decay.values
            lifetimes_dict[label] = lts
        except AttributeError:
            print(f"Warning: 'lifetime_decay' not found in '{label}'. Skipping.")

    # Create DataFrame from the dictionary
    # orient='index' puts the dataset labels as rows
    df = pd.DataFrame.from_dict(lifetimes_dict, orient='index')
    
    # Rename columns to reflect they are lifetimes in ps
    df.columns = [f"Lifetime {i+1} (ps)" for i in range(df.shape[1])]
    df.index.name = "Sample"

    return df
