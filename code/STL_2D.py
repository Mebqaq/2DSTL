import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import multiprocessing as mp
from osgeo import gdal
import os
import concurrent.futures

def process_pixel(args):
    timeseries, kwargs = args
    STL_result = STL(timeseries, **kwargs).fit()
    return STL_result.trend, STL_result.seasonal, STL_result.resid

def STL_2D(TSI: np.ndarray, cpu_count=mp.cpu_count(), **kwargs):
    '''
    Input:
    TSI: A 3D NumPy array where each dimension contains information for (timeslots, columns, rows), respectively.
    cpu_count: Number of CPU processors, default is to call all processors.
    See statsmodels.tsa.seasonal.STL for other parameters (https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.STL.html).

    Output:
    trend_df: DataFrame of trend, indexed by timeslots, each column representing a pixel.
    seasonal_df: DataFrame of season, indexed by timeslots, each column representing a pixel.
    resid_df: DataFrame of residual, indexed by timeslots, each column representing a pixel.
    '''
    timeslots, columns, rows = TSI.shape

    # Prepare arguments for parallel processing
    args = [(TSI[:, i, j], kwargs) for i in range(columns) for j in range(rows)]

    # Use multiprocessing to parallelize the process
    with mp.Pool(cpu_count) as pool:
        results = list(tqdm(pool.imap(process_pixel, args), total=columns*rows))

    trend_list, seasonal_list, resid_list = zip(*results)

    trend_array = np.array(trend_list).reshape(columns, rows, timeslots).transpose(2, 0, 1)
    seasonal_array = np.array(seasonal_list).reshape(columns, rows, timeslots).transpose(2, 0, 1)
    resid_array = np.array(resid_list).reshape(columns, rows, timeslots).transpose(2, 0, 1)

    trend_df = pd.DataFrame(trend_array.reshape(timeslots, -1))
    seasonal_df = pd.DataFrame(seasonal_array.reshape(timeslots, -1))
    resid_df = pd.DataFrame(resid_array.reshape(timeslots, -1))

    return trend_df, seasonal_df, resid_df



def save_image(time, array, rows, columns, output_folder):
    file_name = str(time) + '.tif'
    file_path = os.path.join(output_folder, file_name)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(file_path, columns, rows, 1, gdal.GDT_Float32)
    image = dataset.GetRasterBand(1)
    image.WriteArray(array)
    image.FlushCache()
    dataset = None

def Result2Image(df: pd.DataFrame, rows, columns, output_folder, max_threads=None):
    '''
    df: DataFrame of trend/season/residual/anomaly, which is output by function STL_2D()
    rows, columns: The number of the output images' rows and columns
    output_folder: Folder for output
    max_threads: Maximum number of threads to use (default: None, uses system default)
    
    The names of output images are timeslots
    '''
    
    timeslots = df.shape[1]
    iteration = zip(list(range(timeslots)), df.index.tolist())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for index, time in tqdm(iteration):
            array = df.iloc[index, :].values.reshape(rows, columns)
            futures.append(executor.submit(save_image, time, array, rows, columns, output_folder))
        
        # Ensure all threads complete
        concurrent.futures.wait(futures)

    return


def replace_image_values(time, array, template_image_path, output_folder):
    file_name = str(time) + '.tif'
    file_path = os.path.join(output_folder, file_name)
    
    template_dataset = gdal.Open(template_image_path, gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName('GTiff')
    
    dataset = driver.Create(file_path, template_dataset.RasterXSize, template_dataset.RasterYSize, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(template_dataset.GetGeoTransform())
    dataset.SetProjection(template_dataset.GetProjection())
    
    image = dataset.GetRasterBand(1)
    image.WriteArray(array)
    image.FlushCache()
    dataset = None
    template_dataset = None

def Result2Image_withtemplate(df: pd.DataFrame, template_image_path, output_folder, max_threads=None):
    '''
    df: DataFrame of trend/season/residual/anomaly
    template_image_path: Path to the template image which contains projection and geo-information
    output_folder: Folder for output
    max_threads: Maximum number of threads to use (default: None, uses system default)
    
    The names of output images are timeslots
    '''
    template_dataset = gdal.Open(template_image_path, gdal.GA_ReadOnly)
    rows = template_dataset.RasterYSize
    columns = template_dataset.RasterXSize
    template_dataset = None


    timeslots = df.shape[1]
    iteration = zip(list(range(timeslots)), df.index.tolist())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for index, time in tqdm(iteration, total=df.shape[0]):
            array = df.iloc[index, :].values.reshape(rows, columns)
            futures.append(executor.submit(replace_image_values, time, array, template_image_path, output_folder))
        
        # Ensure all threads complete
        concurrent.futures.wait(futures)

    return



def TukeyAnomalyDetection(df: pd.DataFrame):
    '''
    Detection of mutated residuals.

    input:
    df: DataFrame of residual/trend, which is output by function STL_2D()

    output:
    df_result: Anomaly detection results. -1 for abrupt decrease, 0 for stabilization, and 1 for abrupt increase.
    '''
    
    df_result = df.copy()
    for column_name, column_data in tqdm(df.items(), total=df.shape[1]):
        res_series = column_data.values

        # find the thresholds
        Q1=np.percentile(res_series,25)
        Q3=np.percentile(res_series,75)
        IQR=Q3-Q1
        upper=Q3+1.5*IQR
        lower=Q1-1.5*IQR


        anomaly_flags = np.zeros_like(res_series)
        anomaly_flags[res_series > upper] = 1
        anomaly_flags[res_series < lower] = -1

        df_result[column_name] = anomaly_flags
    return df_result
