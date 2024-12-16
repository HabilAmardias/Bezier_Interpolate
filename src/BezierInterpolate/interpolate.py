from BezierInterpolate.utils import least_square_fit, bezier_curve
import pandas as pd
import numpy as np

def bezier(data:pd.DataFrame | pd.Series, degree:int):
    if not isinstance(data,pd.Series) and not isinstance(data,pd.DataFrame):
        raise Exception('Data should be either Pandas Series or Pandas DataFrame')
    if not isinstance(degree,int) and degree < 0:
        raise ValueError('Degree must be non-negative integer')
    if len(data.shape) > 1 and data.shape[1] > 1:
        raise Exception('Algorithm does not support multivariate data (yet)')
    
    t = np.linspace(0,1,len(data))

    if isinstance(data,pd.DataFrame):
        nan_mask = data.isna().values.reshape(len(data),)
        filled = data[~nan_mask].values.reshape(len(data[~nan_mask],))
    else:
        nan_mask = data.isna().values
        filled = data[~nan_mask].values
    
    
    control_points = least_square_fit(filled,t[~nan_mask],degree)

    if isinstance(data,pd.DataFrame):
        res = pd.DataFrame(bezier_curve(control_points,degree,t),index=data.index,columns=data.columns)
    else:
        res = pd.Series(bezier_curve(control_points,degree,t),index=data.index,name=data.name)

    data = data.fillna(res)

    return data