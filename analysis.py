import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from typing import Tuple

def load_and_clean_data() -> pd.DataFrame:
    """
    Load California Housing dataset and convert it to a pandas DataFrame.
    Returns cleaned DataFrame with relevant features.
    """
    # Load the California housing dataset
    housing = fetch_california_housing()
    
    # Convert to DataFrame
    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )
    df['PRICE'] = housing.target
    
    return df
