import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
from osgeo import gdal
import os

# Processing time series images (TSI) into df
def STL_2D(TSI: np.ndarray,**kwargs):
    '''
    Input:
    TSM: A 3D NumPy array where each dimension contains information for (timeslots,  columns,  rows), respectively.
    See statsmodels.tsa.seasonal.STL for other parameters (https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.STL.html).

    Output:
    trend_df: DataFrame of trend, indexed by timeslots, each column representing a pixel.
    seasonal_df: DataFrame of season, indexed by timeslots, each column representing a pixel.
    resid_df: DataFrame of residual, indexed by timeslots, each column representing a pixel.

    '''
    trend_list = []
    seasonal_list = []
    resid_list = []

    timeslots, rows, columns = TSI.shape
    for i in tqdm(range(columns)):
        for j in range(rows):
            STL_result = STL(TSI[:,i,j], **kwargs).fit()
            
            trend_list.append(STL_result.trend)
            seasonal_list.append(STL_result.seasonal)
            resid_list.append(STL_result.resid)

    trend_df = pd.DataFrame(trend_list, columns=TSI[:,0,0])
    seasonal_df = pd.DataFrame(seasonal_list, columns=TSI[:,0,0])
    resid_df = pd.DataFrame(resid_list, columns=TSI[:,0,0])

    trend_df = trend_df.reset_index(drop=True)
    seasonal_df = seasonal_df.reset_index(drop=True)
    resid_df = resid_df.reset_index(drop=True)

    
    return trend_df, seasonal_df, resid_df


def Result2Image(df: pd.DataFrame, rows, columns, output_folder):
    '''
    df: DataFrame of trend/season/residual/anomaly, which is output by function STL_2D()
    rows, columns: The number of the output images' rows and columns
    output_folder: Folder for output

    The names of output images are timeslots
    '''

    timeslots = df.shape[1]
    iteration = zip(list(range(timeslots)), df.columns.tolist())
    for index, time in tqdm(iteration):
        
        array = df.iloc[:, index].values.reshape(rows, columns)


        file_name = str(time) + '.tif'
        file_path = os.path.join(output_folder, file_name)

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(file_path, columns, rows, 1, gdal.GDT_Float32)
        image = dataset.GetRasterBand(1)
        image.WriteArray(array)
        image.FlushCache()
        dataset = None

    return

def TukeyAnomalyDetection(df: pd.DataFrame):
    '''
    Detection of mutated residuals.

    input:
    df: DataFrame of residual, which is output by function STL_2D()

    output:
    df_result: Anomaly detection results. -1 for abrupt decrease, 0 for stabilization, and 1 for abrupt increase.
    '''
    df_result = df
    for index, row in tqdm(df.iterrows()):
        res_series = row.values

        # find the thresholds
        Q1=np.percentile(res_series,25)
        Q3=np.percentile(res_series,75)
        IQR=Q3-Q1
        upper=Q3+1.5*IQR
        lower=Q1-1.5*IQR


        res_series[((res_series >= lower) & (res_series <= upper))] = 0
        res_series[(res_series > upper)] = 1
        res_series[(res_series < lower)] = -1

        df_result.loc[index] = res_series
    return df_result
