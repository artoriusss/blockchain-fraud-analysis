from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

def plot_category_counts(df, *columns):
    """Plot value counts for specified categorical columns in a DataFrame using Altair"""
    charts = []
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' is not in the DataFrame.")
        
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        chart = alt.Chart(value_counts).mark_bar().encode(
            x=alt.X(f'{column}:N', title=None, sort=alt.EncodingSortField(field='count', order='descending')),
            y=alt.Y('count:Q', title=None),
            tooltip=[alt.Tooltip(f'{column}:N', title='Category'), alt.Tooltip('count:Q', title='Count')]
        ).properties(
            width=300, 
            height=300,
            title=f'Counts for "{column}" column'
        )
        
        charts.append(chart)
    
    final_chart = alt.hconcat(*charts).resolve_scale(y='independent')
    return final_chart

def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix") -> None:
    """Plot a correlation matrix for a DataFrame"""
    corr_matrix = df.corr().reset_index().melt('index')
    corr_matrix = corr_matrix[corr_matrix['index'] >= corr_matrix['variable']]
    corr_matrix.rename(columns={'index': 'y', 'variable': 'x'}, inplace=True)
    corr_matrix['is_diagonal'] = corr_matrix['x'] == corr_matrix['y']

    heatmap = alt.Chart(corr_matrix).mark_rect().encode(
        x=alt.X('x:N', title=''),
        y=alt.Y('y:N', title=''),
        color=alt.condition(
            alt.datum.is_diagonal,
            alt.value('rgba(0, 0, 0, 0.5)'),
            alt.Color('value:Q', scale=alt.Scale(scheme='redpurple'), title='Correlation')
        ),
        tooltip=['x:N', 'y:N', alt.Text('value:Q', format=".3f")]
    ).properties(
        title=title,
        width=400,  
        height=400  
    )

    text = heatmap.mark_text(baseline='middle').encode(
        text=alt.condition(
            alt.datum.is_diagonal,
            alt.value(''),
            alt.Text('value:Q', format=".3f")
        ),
        color=alt.condition(
            alt.datum.value < 0.5,
            alt.value('black'),
            alt.value('white')
        )
    )

    chart = heatmap + text
    chart.display()

def plot_categorical_bar_chart(df: pd.DataFrame, x: str, y: str) -> None:
    aggregated_data = df.groupby([x, y]).size().reset_index(name='count')

    bar_chart = alt.Chart(aggregated_data).mark_bar().encode(
        x=alt.X(x, type='nominal', title=x,
                sort=alt.EncodingSortField(field='count', op='sum', order='descending')),
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color(y, type='nominal', title=y,
                        sort=alt.EncodingSortField(field='count', op='sum', order='descending')),
        tooltip=[x, y, 'count']
    ).properties(
        width=600,
        height=400,
        title=f"Bar Chart of {y} Count by {x}"
    )
    
    bar_chart.display()

def plot_label_distribution_by_comment(df: pd.DataFrame) -> None:
    null_comment_counts = df[df["comment"].isnull()]["label"].value_counts().reset_index()
    null_comment_counts.columns = ['label', 'count']

    non_null_comment_counts = df[~df["comment"].isnull()]["label"].value_counts().reset_index()
    non_null_comment_counts.columns = ['label', 'count']

    pie_chart_null = alt.Chart(null_comment_counts).mark_arc(innerRadius=70).encode(
        theta=alt.Theta(field='count', type='quantitative'),
        color=alt.Color(field='label', type='nominal'),
        tooltip=['label', 'count']
    ).properties(
        title='Labels Distribution (Null Comments)'
    )

    pie_chart_non_null = alt.Chart(non_null_comment_counts).mark_arc(innerRadius=70).encode(
        theta=alt.Theta(field='count', type='quantitative'),
        color=alt.Color(field='label', type='nominal', legend=None),
        tooltip=['label', 'count']
    ).properties(
        title='Labels Distribution (Non-Null Comments)'
    )

    chart = alt.hconcat(pie_chart_null, pie_chart_non_null).resolve_scale(color='independent')
    chart.display()

def plot_categorical_bar_charts(df: pd.DataFrame, xy_pairs) -> None:
    if isinstance(xy_pairs, dict):
        xy_pairs = [xy_pairs]
    
    charts = []
    for pair in xy_pairs:
        x, y = pair['x'], pair['y']
        
        aggregated_data = df.groupby([x, y]).size().reset_index(name='count')

        bar_chart = alt.Chart(aggregated_data).mark_bar().encode(
            x=alt.X(x, type='nominal', title=x,
                    sort=alt.EncodingSortField(field='count', op='sum', order='descending')),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color(y, type='nominal', title=y,
                            sort=alt.EncodingSortField(field='count', op='sum', order='descending')),
            tooltip=[x, y, 'count']
        ).properties(
            width=500,
            height=400,
            title=f"Counts of {y} per {x}"
        )
        charts.append(bar_chart)

    combined_chart = alt.hconcat(*charts).resolve_scale(
        color='independent'
    )
    
    combined_chart.display()