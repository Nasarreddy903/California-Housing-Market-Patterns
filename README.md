# California Housing Data Analysis Project

This project analyzes the California Housing dataset using Python. The analysis includes data cleaning, visualization, and statistical analysis to explore relationships between housing features, with a focus on house prices. This project leverages libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn.

## Project Structure

The project consists of the following Python functions:
- **`load_and_clean_data()`**: Loads the California Housing dataset and prepares it for analysis.
- **`create_distribution_plot()`**: Generates a histogram to visualize the distribution of house prices.
- **`create_scatter_plot()`**: Creates a scatter plot to examine the relationship between median income and house prices.
- **`create_heatmap()`**: Produces a heatmap to show correlations between different housing features.
- **`generate_statistics()`**: Computes descriptive statistics and correlation matrix for further analysis.

## Visualizations

The project includes three primary visualizations:
1. **House Price Distribution** - A histogram showing the distribution of house prices.
2. **Median Income vs. House Prices** - A scatter plot showing the relationship between median income and house prices.
3. **Correlation Heatmap** - A heatmap displaying correlations between various housing features.

These visualizations are saved as `.png` files upon execution.

## Requirements

- Python 3.7+
- The following Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

You can install the required libraries with:

    pip install pandas numpy matplotlib seaborn scikit-learn

## Usage

1. Clone the repository:
   
       git clone https://github.com/Nasarreddy903/California-Housing-Market-Patterns
       cd California-Housing-Market-Patterns

2. Run the main script to generate visualizations and statistics:
   
       python analysis.py

3. The script will display descriptive statistics and correlation matrices in the console, and it will save the visualizations as PNG files in the project directory.

## Code Walkthrough

Here’s a quick overview of each function:

- **`load_and_clean_data()`**: Loads the California Housing dataset from scikit-learn and converts it into a DataFrame, adding relevant column names.
  
- **`create_distribution_plot(df: pd.DataFrame)`**: Takes a DataFrame and generates a histogram of house prices.
  
- **`create_scatter_plot(df: pd.DataFrame)`**: Creates a scatter plot comparing median income with house prices, helping visualize any correlation.
  
- **`create_heatmap(df: pd.DataFrame)`**: Computes a correlation matrix and displays it as a heatmap for feature comparison.
  
- **`generate_statistics(df: pd.DataFrame)`**: Produces descriptive statistics and a correlation matrix, useful for understanding feature relationships.

## Example Output

After running the script, you will see the following output files in the project directory:
- distribution_plot.png
- scatter_plot.png
- heatmap.png

## Contributing

If you’d like to contribute to this project, feel free to open issues and pull requests. Contributions are always welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project uses the California Housing dataset available in the Scikit-Learn library. Special thanks to the open-source contributors who developed and maintain these Python libraries!

---
