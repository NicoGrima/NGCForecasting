import numpy as np
import pandas as pd
import torch
import numpy as np

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
    continuous_labels = pd.DataFrame(columns = ["ElapsedTime", "Workload"])
    for index, row in df.iterrows():
        if row.workload_classification == 100:
            # If overload then we predict 100 (with lower confidence values lowering the value)
            continuous_labels.loc[len(continuous_labels)] = [
                row.ElapsedTime,
                row.workload_classification+(row.workload_confidence*100-100)/2
            ]
        else:
            # If underload then we predict 0 (with lower confidence values increasing the value)
            # If optimal then we predict 33 (with lower confidence values increasing the value)
            continuous_labels.loc[len(continuous_labels)] = [
                row.ElapsedTime,
                row.workload_classification+(100-row.workload_confidence*100)/2
            ]
    return continuous_labels, indices

def normalize(seq_data):
    # Get the mean and standard deviation of each column
    sequence_mean = seq_data.mean().to_numpy()
    sequence_std = seq_data.std()
    sequence_std = sequence_std.fillna(np.float32(0.0))
    sequence_std = sequence_std.to_numpy()

    # Calculate the z-score of each column
    sequence_norm = (seq_data - sequence_mean)/sequence_std

    # Replace NAN z-scores with 0
    sequence_norm = sequence_norm.fillna(np.float32(0.0))
    sequence_norm = sequence_norm.replace([np.inf, -np.inf], np.float32(0.0))
    sequence_norm = sequence_norm.to_numpy()

    return np.expand_dims(np.array(sequence_norm), axis=0), np.array(sequence_mean), np.array(sequence_std)

def forecast(model, input_tensor, token_tensor, model_to_load):
    output = None
    if model_to_load == 'lstm':
        output = model.forward(input_tensor)
    elif model_to_load == 'transformer':
        label_length = 100  # update based on desired/trained label length
        dec_tensor = token_tensor
        for prediction in range(label_length):
            output = model.forward(input_tensor, dec_tensor)
            dec_tensor = torch.cat((token_tensor, output.squeeze(2)), dim=1)
    else:
        raise ValueError('No such model is available')

    return output

def format_data(df_light, df_hemodynamics, df_classified):
    # Create copies of all input data
    df_light = df_light.copy(deep = True)
    df_hemodynamics = df_hemodynamics.copy(deep = True)
    df_classified = df_classified.copy(deep = True)

    # Trim datasets
    if len(df_light) >= 500:
        df_light = df_light.iloc[-500:, :]
    if len(df_hemodynamics) >= 500:
        df_hemodynamics = df_hemodynamics.iloc[-500:, :]
    if len(df_classified) >= 500:
        df_simple_linear = df_classified.iloc[-500:, :]

    # Convert workload to continuous measure
    df_simple_linear, _ = continuous_simple(df_classified)

    # Down-select to only samples that include all data
    valid_timestamps = set.intersection(
        set(df_light.ElapsedTime),
        set(df_hemodynamics.ElapsedTime),
        set(df_simple_linear.ElapsedTime))
    df_light = df_light.loc[df_light.ElapsedTime.isin(valid_timestamps)]
    df_hemodynamics = df_hemodynamics.loc[df_hemodynamics.ElapsedTime.isin(valid_timestamps)]
    df_simple_linear = df_simple_linear.loc[df_simple_linear.ElapsedTime.isin(valid_timestamps)]

    # Combine datasets into expected format
    df = pd.merge(
        df_simple_linear,
        pd.merge(
            df_light,
            df_hemodynamics,
            on = ["ElapsedTime"]),
        on = ["ElapsedTime"])
    df = df.drop(columns = ["ElapsedTime"])
    return df