import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt


def continuous_simple(df):
    # Check if too many Nan values in workload confidence
    mask = df['workload_confidence'].isnull()
    indices = df[mask].index

    # Drop Nan values in confidence
    df = df.drop(indices)

    # Switch string values with ints
    map_dict = {'Underload': 0, 'Optimal': 33, 'Overload': 100}
    df['workload_classification'] = df['workload_classification'].map(map_dict)

    # Create continuous labels
    continuous_labels = []
    for index, row in df.iterrows():
        if df.iloc[index, 0] == 100:
            # If overload then we predict 100 (with lower confidence values lowering the value)
            continuous_labels.append(df.iloc[index, 0]+(df.iloc[index, 1]*100-100)/2)
        else:
            # If underload then we predict 0 (with lower confidence values increasing the value)
            # If optimal then we predict 33 (with lower confidence values increasing the value)
            continuous_labels.append(df.iloc[index, 0]+(100-df.iloc[index, 1]*100)/2)
    return np.array(continuous_labels), indices


def continuous_complex(df):
    # Need to manually check if too many Nan values in confidence might affect the data
    mask = df['workload_confidence'].isnull()
    indices = df[mask].index

    # Switch string values with ints
    map_dict = {'Underload': 0, 'Optimal': 50, 'Overload': 100}
    df['workload_classification'] = df['workload_classification'].map(map_dict)

    # Create continuous labels
    continuous_labels = []
    last_load = None
    for index, row in df.iterrows():

        # Track the last workload (underload, optimal, overload)
        if index != 0 and df.iloc[index, 0] != df.iloc[index-1, 0]:
            last_load = df.iloc[index-1, 0]

        if df.iloc[index, 0] == 100:
            continuous_labels.append(df.iloc[index, 0] + (df.iloc[index, 1]*100 - 100) / 2)
        elif df.iloc[index, 0] == 50:
            if last_load == 0:
                # If the last workload value was underload then lower confidence means a value under 50
                continuous_labels.append(df.iloc[index, 0] + (df.iloc[index, 1]*100 - 100) / 4)
            elif last_load == 100:
                # If the last workload value was overload then lower confidence means a value over 50
                continuous_labels.append(df.iloc[index, 0] + (100 - df.iloc[index, 1]*100) / 4)
        else:
            continuous_labels.append(df.iloc[index, 0] + (100 - df.iloc[index, 1]*100) / 3)

    return np.array(continuous_labels), indices


def graph_figures(df_cat, df_linear):
    x = np.arange(df_linear.shape[0])
    cat = df_cat['workload_classification'].dropna()
    map_dict = {'Underload': 0, 'Optimal': 50, 'Overload': 100}
    cat = cat.map(map_dict).values
    plt.plot(x, df_linear, label='Continuous')
    plt.plot(x, cat, label='Categorical')
    plt.xlabel('Time')
    plt.ylabel('Workload')
    plt.title('Linear Transformation of Workload')
    plt.legend(fontsize=12, loc='lower left')
    plt.show()


def readFile_sqlite(file_name: str, transformation: str, graph_comparison=False):
    # Connect to the SQLite database file
    conn = sqlite3.connect(file_name)

    tables = {
        'classified': 'ocarina_biomeasures_classified_hemodynamics',
        'light': 'ocarina_biomeasures_light_intensities',
        'markers': 'ocarina_biomeasures_markers',
        'hemodynamics': 'ocarina_biomeasures_processed_hemodynamics'
    }

    # Query the database and fetch the results into a Pandas dataframe
    df_classified = pd.read_sql_query("SELECT * FROM ocarina_biomeasures_classified_hemodynamics", conn)
    df_light = pd.read_sql_query("SELECT * FROM ocarina_biomeasures_light_intensities", conn)
    df_hemodynamics = pd.read_sql_query("SELECT * FROM ocarina_biomeasures_processed_hemodynamics", conn)

    # any_null_ = df_light.isna().any(axis=1)
    # first = df_light[any_null_]
    # any_null_rows = df_hemodynamics.isna().any(axis=1)
    # sec = df_hemodynamics[any_null_rows]

    # Trim data
    df_classified = df_classified.iloc[:, 3:]
    df_light = df_light.iloc[:, :-3]
    df_hemodynamics = df_hemodynamics.iloc[:, 3:]

    # Turn workload classification into a continuous variable (simple vs complex will determine the
    # type of linear transformation)
    if transformation == 'simple':
        df_simple_linear, indices = continuous_simple(df_classified)
        # If graph comparison is true then we graph the ordinal vs continuous data
        if graph_comparison:
            graph_figures(df_classified, df_simple_linear)
        # Concatenate the data with the classification
        df = pd.concat([pd.DataFrame(df_simple_linear, columns=['Workload']), df_light, df_hemodynamics], axis=1)
        df = df.drop(indices)

    elif transformation == 'complex':
        df_complex_linear, indices = continuous_complex(df_classified)
        if graph_comparison:
            graph_figures(df_classified, df_complex_linear)
        df = pd.concat([pd.DataFrame(df_complex_linear, columns=['Workload']), df_light, df_hemodynamics], axis=1)
        df = df.drop(indices)

    else:
        raise ValueError('Not an acceptable transformation. Supported transformations: "simple" or "complex"')

    return df

def readFile_pkl(file_name: str):

    df = pd.read_pickle(file_name)
    wkld = df.iloc[:, -3:]
    last_target = df.columns.get_loc('Opt16_Oxy3')
    first_target = df.columns.get_loc('Opt1_HbR')
    df = df.iloc[:, first_target:last_target+1]
    return df

