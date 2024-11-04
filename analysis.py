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

def create_scatter_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Create a scatter plot showing relationship between median income
    and house prices.
    Returns matplotlib figure object.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, 
                    x='MedInc', 
                    y='PRICE',
                    alpha=0.5)
    plt.title('Median Income vs House Prices')
    plt.xlabel('Median Income (10k USD)')
    plt.ylabel('House Price (100k USD)')
    return plt.gcf()


def create_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Create a correlation heatmap of housing metrics.
    Returns matplotlib figure object.
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1,
                center=0,
                fmt='.2f')
    plt.title('Correlation Heatmap of Housing Features')
    return plt.gcf()

def generate_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate descriptive statistics and correlation matrix.
    Returns tuple of (descriptive stats, correlation matrix).
    """
    descriptive_stats = df.describe()
    correlation_matrix = df.corr()
    
    return descriptive_stats, correlation_matrix


def main():
    # Set the style for better-looking plots
    plt.style.use('seaborn')
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Generate all visualizations
    dist_plot = create_distribution_plot(df)
    scatter_plot = create_scatter_plot(df)
    heatmap = create_heatmap(df)
    
    # Generate statistics
    desc_stats, corr_matrix = generate_statistics(df)
    
    # Save plots
    dist_plot.savefig('distribution_plot.png', bbox_inches='tight', dpi=300)
    scatter_plot.savefig('scatter_plot.png', bbox_inches='tight', dpi=300)
    heatmap.savefig('heatmap.png', bbox_inches='tight', dpi=300)
    
    # Print statistics
    print("\nDescriptive Statistics:")
    print(desc_stats)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    plt.close('all')  # Clean up memory

if __name__ == "__main__":
    main()
