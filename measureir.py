import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from tqdm import tqdm

# Metric Definitions and Mathematical Expressions
METRICS_INFO = {
    "Mean Return": {
        "description": "The average daily return of a stock over the selected period.",
        "formula_discrete": r"\mu = \frac{1}{T} \sum_{t=1}^T R_t",
        "formula_continuous": r"\mu = \int_{-\infty}^{\infty} r \, f(r) \, dr",
    },
    "Variance": {
        "description": "Measures the dispersion of returns from the mean.",
        "formula_discrete": r"\sigma^2 = \frac{1}{T} \sum_{t=1}^T (R_t - \mu)^2",
        "formula_continuous": r"\sigma^2 = \int_{-\infty}^{\infty} (r - \mu)^2 \, f(r) \, dr",
    },
    "Semi-Variance": {
        "description": "Measures the variance of only negative returns.",
        "formula_discrete": r"\text{SemiVariance} = \frac{1}{T} \sum_{t=1}^T (R_t - \mu)^2, \text{ if } R_t < \mu",
        "formula_continuous": r"\text{SemiVariance} = \int_{-\infty}^{\mu} (r - \mu)^2 \, f(r) \, dr",
    },
    "Shortfall Probability": {
        "description": "The probability that the return is negative.",
        "formula_discrete": r"P(R_t < 0)",
        "formula_continuous": r"P(R < 0) = \int_{-\infty}^{0} f(r) \, dr",
    },
    "VaR (95%)": {
        "description": "Value at Risk at a 95% confidence level. The threshold below which 5% of returns lie.",
        "formula_discrete": r"\text{VaR}_{95\%} = \text{Percentile}(R, 5\%)",
        "formula_continuous": r"\int_{-\infty}^{\text{VaR}_{95\%}} f(r) \, dr = 0.05",
    },
    "TVaR (95%)": {
        "description": "Tail Value at Risk, the average return below the 95% VaR.",
        "formula_discrete": r"\text{TVaR}_{95\%} = \frac{\sum_{R_t < \text{VaR}_{95\%}} R_t}{N}",
        "formula_continuous": r"\text{TVaR}_{95\%} = \frac{\int_{-\infty}^{\text{VaR}_{95\%}} r \, f(r) \, dr}{P(R \leq \text{VaR}_{95\%})}",
    },
    "Sharpe Ratio": {
        "description": "Measures risk-adjusted return using total variance.",
        "formula_discrete": r"\text{Sharpe Ratio} = \frac{\mu - r_f}{\sigma}",
        "formula_continuous": r"\text{Sharpe Ratio} = \frac{\int_{-\infty}^{\infty} (r - r_f) \, f(r) \, dr}{\sqrt{\int_{-\infty}^{\infty} (r - \mu)^2 \, f(r) \, dr}}",
    },
    "Sortino Ratio": {
        "description": "Measures risk-adjusted return using downside risk (semi-variance).",
        "formula_discrete": r"\text{Sortino Ratio} = \frac{\mu - r_f}{\sqrt{\text{SemiVariance}}}",
        "formula_continuous": r"\text{Sortino Ratio} = \frac{\int_{-\infty}^{\infty} (r - r_f) \, f(r) \, dr}{\sqrt{\int_{-\infty}^{\mu} (r - \mu)^2 \, f(r) \, dr}}",
    },
}



# Function to calculate metrics
def calculate_metrics(prices):
    """
    Calculate investment return metrics based on daily stock prices.
    """
    returns = prices.pct_change().dropna()
    if len(returns) == 0:
        return None, None, None, None, None, None, None, None

    mean_return = returns.mean()
    variance = returns.var()
    semi_variance = returns[returns < 0].var()
    shortfall_prob = (returns < 0).mean()
    var_95 = np.percentile(returns, 5)
    tvar_95 = returns[returns <= var_95].mean()
    sharpe_ratio = mean_return / np.sqrt(variance) if variance > 0 else None
    sortino_ratio = mean_return / np.sqrt(semi_variance) if semi_variance > 0 else None

    return mean_return, variance, semi_variance, shortfall_prob, var_95, tvar_95, sharpe_ratio, sortino_ratio


# Function to load data and compute yearly metrics
def load_and_compute_metrics(data_dir):
    datanames = os.listdir(data_dir)
    all_results = pd.DataFrame()

    for dataname in tqdm(datanames, desc="Processing Stocks"):
        results = []
        stock = dataname.split(".")[0]
        try:
            datamain = pd.read_csv(f"{data_dir}/{dataname}", index_col=0, parse_dates=True)["Close"]
            for year in range(int(datamain.index[0].year), 2025):
                try:
                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"
                    data = datamain[(datamain.index >= start_date) & (datamain.index <= end_date)]

                    if len(data) < 10:
                        continue
                    
                    metrics = calculate_metrics(data)
                    if metrics[0] is not None:
                        results.append([stock, year] + list(metrics))
                except:
                    continue

            # Append results to the main DataFrame
            columns = ["Stock", "Year", "Mean Return", "Variance", "Semi-Variance", 
                       "Shortfall Probability", "VaR (95%)", "TVaR (95%)", "Sharpe Ratio", "Sortino Ratio"]
            stock_results = pd.DataFrame(results, columns=columns)
            all_results = pd.concat([all_results, stock_results])
        except:
            continue

    all_results.set_index(['Stock', 'Year'], inplace=True)
    return all_results


# Streamlit app
st.title("ST2109: MEASURE OF INVESTMENT RISK (ASSIGNMENT 2)")
st.sidebar.title("Options")

DATA_DIR = "./Data"

# Display metric definitions
st.header("Metric Definitions and Mathematical Expressions")
for metric, info in METRICS_INFO.items():
    st.subheader(metric)
    st.markdown(f"**Definition:** {info['description']}")
    st.markdown("**Discrete Formula:**")
    st.latex(info["formula_discrete"])
    st.markdown("**Continuous Formula:**")
    st.latex(info["formula_continuous"])



# Load and process stock data
if not os.path.exists(DATA_DIR):
    st.error("Data directory does not exist. Please ensure stock data is saved in './Data/'.")
else:
    st.sidebar.write("Loading data...")
    metrics_data = load_and_compute_metrics(DATA_DIR)

    # Sidebar controls
    all_stocks = metrics_data.index.get_level_values('Stock').unique()
    selected_stocks = st.sidebar.multiselect("Select Stocks for Analysis", all_stocks, default=list(all_stocks)[:5])

    selected_metrics = st.sidebar.multiselect(
        "Select Metrics for Analysis",
        ["Mean Return", "Variance", "Semi-Variance", "Shortfall Probability", "VaR (95%)", 
         "TVaR (95%)", "Sharpe Ratio", "Sortino Ratio"],
        default=["Mean Return"]
    )

    # EDA Section
    st.header("Exploratory Data Analysis (EDA)")
    
    # Correlation Matrix for Selected Stocks
    st.subheader("Correlation Matrix")
    corr_data = metrics_data.loc[selected_stocks, "Mean Return"].unstack(level=0).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix of Mean Returns")
    st.pyplot(fig)

    # Boxplot of Mean Returns Across Selected Stocks
    st.subheader("Distribution of Mean Returns Across Stocks")
    boxplot_data = metrics_data.loc[selected_stocks, "Mean Return"].reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Stock", y="Mean Return", data=boxplot_data, ax=ax)
    ax.set_title("Mean Return Distribution Across Stocks")
    ax.set_ylabel("Mean Return")
    ax.set_xlabel("Stock")
    st.pyplot(fig)

    # Comparative Analysis Across Stocks
    st.header("Comparative Analysis Across Stocks")
    for metric in selected_metrics:
        st.subheader(f"Comparative Analysis for {metric}")
        comp_data = metrics_data.reset_index().pivot(index="Year", columns="Stock", values=metric)
        comp_data = comp_data[selected_stocks]  # Filter for selected stocks
        st.line_chart(comp_data)

    # Pairplot of Selected Metrics for All Selected Stocks
    st.subheader("Pairwise Relationships of Metrics")
    pairplot_data = metrics_data.reset_index()
    pairplot_data = pairplot_data[pairplot_data["Stock"].isin(selected_stocks)]
    selected_pair_metrics = st.sidebar.multiselect(
        "Select Metrics for Pairwise Analysis",
        ["Mean Return", "Variance", "Semi-Variance", "Shortfall Probability", "VaR (95%)", 
         "TVaR (95%)", "Sharpe Ratio", "Sortino Ratio"],
        default=["Mean Return", "Sharpe Ratio"]
    )
    if len(selected_pair_metrics) > 1:
        fig = sns.pairplot(pairplot_data, vars=selected_pair_metrics, hue="Stock")
        st.pyplot(fig)

    # Download Processed Metrics
    csv = metrics_data.to_csv()
    st.download_button(
        label="Download Metrics as CSV",
        data=csv,
        file_name="investment_metrics.csv",
        mime="text/csv",
    )
