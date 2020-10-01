"""
DESCR: Contain a few helpers functions to keep notebook code cleaner
AUTHOR Retima Dangol
"""

# IMPORT STATEMENTS
import pandas as pd
import numpy as np
import scipy.stats as ss
from IPython.core.display import HTML


def css_styling(css_file_path="./custom.css"):
    """
    DESCR: return a loaded css file html object to render in a notebook
    """
    styles = open(css_file_path, "r").read()
    return HTML(styles)


def cramers_v(x, y):
    """
    DESCR: get correlation between 2 categorical series
    Credit to Shaked Zychlinski
    Source: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def cat_correlation_heatmap(df):
    """
    Apply cramers_v to a dataframe to get a correlation map
    this correlation map can then be plotted easier
    """
    cat_columns = df.select_dtypes(exclude=["number"]).columns
    subset = df[cat_columns]
    size = len(cat_columns)
    corr_vals = np.zeros((size, size))
    for ind1, col1 in enumerate(cat_columns):
        for ind2, col2 in enumerate(cat_columns):
            corr = cramers_v(df[col1], df[col2])
            corr_vals[ind1][ind2] = corr
    return corr_vals, cat_columns
