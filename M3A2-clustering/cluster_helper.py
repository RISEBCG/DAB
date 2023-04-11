import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

from google.colab import widgets

import warnings
warnings.filterwarnings('ignore')

# Creating required Python functions for this session
# Press Shift + Enter


def standardize_numeric_data(df, numeric_columns):
    df_original = df.copy()
    df[numeric_columns] = MinMaxScaler().fit_transform(df[numeric_columns])

    # Create box plots for the original and normalized data
    original = px.box(df_original, y=numeric_columns)
    normalised = px.box(df, y=numeric_columns)

    # Create a 1x2 grid of subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Original Data', 'Normalized Data'])

    # Add the box plots to the subplots
    fig.add_trace(original['data'][0], row=1, col=1)
    fig.add_trace(normalised['data'][0], row=1, col=2)

    fig.update_layout(title='Data scale comparison before and after standardization')

    fig.show()

    return df, df_original


def run_kmeans_model(cluster_count, df, df_original, numeric_columns):
    km_estimator = KMeans(n_clusters=cluster_count, random_state=42)
    df_labels = km_estimator.fit_predict(df[numeric_columns])

    df_pred = df_original.copy()
    df_pred['kmeans_label'] = df_labels
    return df_pred


def create_segment_profiling_chart(df, cluster):
    df = df[df['kmeans_label'] == cluster]
    fig = make_subplots(rows=2, cols=3,
                        specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "treemap"}], [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]],
                        subplot_titles=("Age: Avg at {:.2f}".format(df['age'].mean()),
                                        "Income: Avg at {:.2f}".format(df['annual_income'].mean()),
                                        "SKU Purchased",
                                        "Total Net Sales: Avg at {:.2f}".format(df['total_net_sales'].mean()),
                                        "Days Since Last Purchased: Avg at {:.2f}".format(df['most_recent_purchase_days'].mean()),
                                        "Total No. Transactions: Avg at {:.2f}".format(df['total_no_transactions'].mean())))

    def get_tree_fig(df):
        treemap_data = df[['Greek yogurt_count', 'Nuts_count', 'Apple slices_count', 'Chips_count', 'Soda_count', 'Chocolate bar_count', 'Seasonal special_count', 'Promotion item_count', 'Lunchbox - Beef_count', 'Lunchbox - Vegan_count']].sum().reset_index()
        treemap_data.columns = ['category', 'value']
        fig = go.Figure(go.Treemap(
            labels=treemap_data['category'],
            parents=[""] * len(treemap_data),
            values=treemap_data['value']))
        return fig

    age_trace = go.Histogram(x=df['age'], showlegend=False)
    income_trace = go.Histogram(x=df['annual_income'], showlegend=False)
    tree_trace = get_tree_fig(df)
    monetary_trace = go.Histogram(x=df['total_net_sales'], showlegend=False)
    recency_trace = go.Histogram(x=df['most_recent_purchase_days'], showlegend=False)
    frequency_trace = go.Histogram(x=df['total_no_transactions'], showlegend=False)

    fig.add_trace(age_trace, row=1, col=1)
    fig.add_trace(income_trace, row=1, col=2)
    fig.add_trace(tree_trace.data[0], row=1, col=3)
    fig.add_trace(monetary_trace, row=2, col=1)
    fig.add_trace(recency_trace, row=2, col=2)
    fig.add_trace(frequency_trace, row=2, col=3)

    fig.update_layout(title='Customer Profile for Customer Segment Number {}. Segment Size: {}'.format(cluster, len(df)))
    fig.show()


def plot_elbow_chart(df, numeric_columns):
    sse = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(df[numeric_columns])
        sse.append(km.inertia_)

    plt.plot(range(1, 10), sse, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow chart')
    plt.show()


def create_cluster_visualisation(df_pred):
    x_range = np.percentile(df_pred['most_recent_purchase_days'], [2.5, 97.5])
    y_range = np.percentile(df_pred['total_net_sales'], [2.5, 97.5])
    z_range = np.percentile(df_pred['total_no_transactions'], [2.5, 97.5])

    fig = go.Figure(data=go.Scatter3d(
        x=df_pred['most_recent_purchase_days'],
        y=df_pred['total_net_sales'],
        z=df_pred['total_no_transactions'],
        mode='markers',
        marker=dict(
            color=df_pred['kmeans_label'],
            colorscale='Viridis',
            opacity=0.8,
            size=5
        ),
        showlegend=False
    ))

    fig.update_layout(scene=dict(
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        zaxis=dict(range=z_range),
        xaxis_title='Recency',
        yaxis_title='Monetary',
        zaxis_title='Frequency'
    ))
    fig.show()


def experiment_different_cluster(df, df_original, cluster_columns, number_of_clusters):
    df_pred = run_kmeans_model(number_of_clusters, df, df_original, cluster_columns)

    tab_cols = ['Cluster Visualisation']
    for i in range(number_of_clusters):
        tab_cols.append("Cluster {}".format(i))

    t = widgets.TabBar(tab_cols)
    with t.output_to(0):
        create_cluster_visualisation(df_pred)

    for i in range(number_of_clusters):
        with t.output_to(i + 1):
            create_segment_profiling_chart(df_pred, i)
