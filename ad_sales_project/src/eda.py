"""
eda.py
------
Performs exploratory data analysis (EDA) on the cleaned advertising dataset.
Generates and saves plots, and writes insights to a markdown file.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_cleaned_data(path):
    return pd.read_csv(path)

def plot_correlation_heatmap(df, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_pairplot(df, out_path):
    sns.pairplot(df)
    plt.suptitle('Pairplot of Features', y=1.02)
    plt.savefig(out_path)
    plt.close()

def plot_feature_vs_sales(df, out_dir):
    features = ['TV', 'Radio', 'Newspaper']
    paths = []
    for feat in features:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[feat], y=df['Sales'])
        plt.title(f'{feat} vs Sales')
        plt.xlabel(feat)
        plt.ylabel('Sales')
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'{feat}_vs_Sales.png')
        plt.savefig(out_path)
        plt.close()
        paths.append(out_path)
    return paths

def generate_eda_insights(df, out_path):
    corr = df.corr()
    insights = []
    insights.append('# EDA Insights')
    insights.append('\n## Correlation Matrix')
    insights.append(corr.to_markdown())
    # Key findings
    insights.append('\n## Key Findings')
    if corr['Sales']['TV'] > corr['Sales']['Radio'] and corr['Sales']['TV'] > corr['Sales']['Newspaper']:
        insights.append('- TV advertising budget has the strongest positive correlation with Sales.')
    if corr['Sales']['Radio'] > 0.3:
        insights.append('- Radio budget also shows a moderate positive correlation with Sales.')
    if corr['Sales']['Newspaper'] < 0.2:
        insights.append('- Newspaper budget has a weak or negligible correlation with Sales.')
    with open(out_path, 'w') as f:
        f.write('\n'.join(insights))

def run_eda():
    data_path = os.path.join('..', 'data', 'cleaned_advertising.csv')
    fig_dir = os.path.join('..', 'reports', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    insights_path = os.path.join('..', 'reports', 'eda_insights.md')
    df = load_cleaned_data(data_path)
    plot_correlation_heatmap(df, os.path.join(fig_dir, 'correlation_heatmap.png'))
    plot_pairplot(df, os.path.join(fig_dir, 'pairplot.png'))
    plot_feature_vs_sales(df, fig_dir)
    generate_eda_insights(df, insights_path)
    print(f"EDA complete. Figures saved to {fig_dir} and insights to {insights_path}")

if __name__ == "__main__":
    run_eda()
