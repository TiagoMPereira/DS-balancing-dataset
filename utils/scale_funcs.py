import pandas as pd

data = {'col1': [-2.5, -1.8, 0.2, 1.4, 3.7], 'col2': ['str1', 'str2', 4.2, -1.7, 2.9], 'col3': [True, False, True, False, True]}
df = pd.DataFrame(data)

def scale_column(df, col, scaler, scale_positive):
  if pd.api.types.is_numeric_dtype(df[col]):
    positive_values = df.loc[df[col] >= 0, col]
    negative_values = df.loc[df[col] < 0, col]
    
    # Check if there are values to scale (positive or negative)
    if (scale_positive and not positive_values.empty) or (not scale_positive and not negative_values.empty):
      if scale_positive:
        scaled_values = scaler.fit_transform(positive_values.values.reshape(-1, 1))
      else:
        scaled_values = scaler.fit_transform(negative_values.values.reshape(-1, 1))
      df.loc[(df[col] >= 0) if scale_positive else (df[col] < 0), col] = scaled_values.squeeze()
  return df

def scale_positives(df, scaler):
  for col in df.select_dtypes(include='number'):
    df = scale_column(df.copy(), col, scaler, scale_positive=True)
  return df

def scale_negatives(df, scaler):
  for col in df.select_dtypes(include='number'):
    df = scale_column(df.copy(), col, scaler, scale_positive=False)
  return df

def scale_to_minus_one_plus_one(df):
  # Select numeric columns
  numeric_cols = df.select_dtypes(include='number')

  # Scale each numeric column
  for col in numeric_cols:
    # Find minimum and maximum (absolute values)
    min_value = min(df[col])
    max_value = max(df[col])
    abs_min = abs(min_value)
    abs_max = abs(max_value)

    # Separate negative and positive values (boolean mask)
    negative_mask = df[col] < 0
    positive_mask = df[col] >= 0

    # Invert and scale negative values
    if any(negative_mask):
      scaled_negative_values = [(abs_min - value) / abs_min for value in df.loc[negative_mask, col]]
      df.loc[negative_mask, col] = scaled_negative_values

    # Scale positive values
    if any(positive_mask):
      scaled_positive_values = [1 - (value / abs_max) for value in df.loc[positive_mask, col]]
      df.loc[positive_mask, col] = scaled_positive_values

  return df

def add_bin_columns(df, bins, col_prefix="bin"):
  for col in df.columns:
    bins_reshaped = bins.reshape(-1, 1)  # Reshape to column vector
    df[col_prefix + "_" + col] = bins_reshaped.copy()  # Avoid modifying original bins
  return df
