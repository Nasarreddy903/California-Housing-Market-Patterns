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

def create_distribution_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Create a histogram showing the distribution of house prices.
    Returns matplotlib figure object.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='PRICE', bins=50)
    plt.title('Distribution of House Prices in California')
    plt.xlabel('Price (100k USD)')
    plt.ylabel('Count of Houses')
    return plt.gcf()
