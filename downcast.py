import pandas as pd
import numpy as np
import warnings

def downcast_df(df, categorical_cols=(), na_suffix="_na", verbose="none"):
    """
    parameters
    verbose = None, "dataframe", "columns"
    """
    
    
    """
    Casts dataframe column to most compact types.
    Args:
        df: Dataframe to be compressed.
        categorical_cols: Iterable of categorical column names.
        na_col_suffix: Suffix of bool columns indicating nans from converted integer columns
        verbose: How much information to output
    Returns: 
        out: input dataframe appeneded with vector for bag of words
        bow_reference: dataframe with bag of words index
    """
#   If max_value exceeds half of integer range, use type one larger
    int_ranges = {
        (-2**6 , 2**6) :np.int8,
        (-2**14, 2**14):np.int16,
        (-2**30, 2**30):np.int32,
        (-2**62, 2**62):np.int64,
    }
    
    na_cols = []
    
    start_size = df.memory_usage(deep=True).sum()//(1024)**2

    for c in df.columns:
        if verbose=="columns":
            log_string = f"Column '{c}' dtype: {df[c].dtype} -> "
        
        # if column is categorical
        if c in categorical_cols:
            df[c] = df[c].astype("category")
            if verbose=="columns":
                print(log_string + str(df[c].dtype))
            continue
            
        IsInt = False
        mx = df[c].max()
        mn = df[c].min()

        # test if column can be converted to an integer
        try:
            asint = df[c].fillna(0).astype(np.int64)
            result = (df[c] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
#       If column is string
        except ValueError as e:
            if verbose=="columns":
                print(log_string + str(df[c].dtype))
            continue
        
        
        if IsInt and np.isfinite(mn) and np.isfinite(mx):
            for int_range in int_ranges:
                if int_range[0] <= mn and mx <= int_range[1]:
                    if df[c].isna().sum()>0:
                        na_cols.append(c+na_suffix)
                        df[c+na_suffix] = df[c].isna().astype(bool)
                        df[c] = df[c].fillna(1+int_range[0]*2)
                    df[c] = df[c].astype(int_ranges[int_range])
                    break
        else:
            df[c] = df[c].astype("float16")
            
        if verbose=="columns":
            print(log_string + str(df[c].dtype))
    
    if verbose != "none":
        print(f"Memory usage of df before: {start_size} MB")
        print(f"Memory usage of df after: {df.memory_usage(deep=True).sum()//(1024)**2} MB")
    if len(na_cols) > 0:
        warnings.warn(f"Number of integer columns with NA values:{len(na_cols)}")
    
    return df, na_cols
