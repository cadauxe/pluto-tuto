import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.plot import show


def get_file_size_in_mb(file_path):
    """
    Return a str containing a file size in MB
    """
    raw_size = Path(file_path).stat().st_size
    return raw_size * 1e-6


def compare_read_write_times(read_times, write_times, labels):
    data = pd.DataFrame({
        'Formats': labels,
        'Read': read_times,
        'Write': write_times
    })

    # group times by format
    data_long = data.melt(id_vars='Formats', var_name='Type', value_name='Time (s)')
    plot = data_long.hvplot.bar(
        x='Formats',
        y='Time (s)',
        by='Type',
        title="Read and write times, depending on the format",
        xlabel='Formats',
        ylabel='Time (s)',
        color=['#a8dadc', '#457b9d']
    )
    return plot


def download_sample_data(download_dir):
    output_raster = Path(
        download_dir) / "data" / "xt_SENTINEL2B_20180621-111349-432_L2A_T30TWT_D_V1-8_RVBPIR.tif"

    if output_raster.exists():
        return output_raster

    zip_file_url = "https://www.orfeo-toolbox.org/packages/WorkshopData/data_otb-guided-tour.zip"
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(download_dir)

    assert output_raster.exists()

    (Path(
        download_dir) / "data" / "xt_SENTINEL2B_20180701-111103-470_L2A_T30TWT_D_V1-8_RVBPIR.tif").unlink()
    (Path(
        download_dir) / "data" / "xt_SENTINEL2A_20180706-110918-241_L2A_T30TWT_D_V1-8_RVBPIR.tif").unlink()
    (Path(
        download_dir) / "data" / "xt_SENTINEL2B_20180711-111139-550_L2A_T30TWT_D_V1-8_RVBPIR.tif").unlink()

    return output_raster


def show_raster(input_image: str):
    """
    Displays the RGB bands of a raster image with rescaling for visualization.

    Parameters
    ----------
    input_image : str
        Path to the input raster image.
    """
    with rasterio.open(input_image) as src:
        # get the RGB bands
        raster_data = src.read()
        raster_data = raster_data[0:3, :, :]

        # create new range values for visualization purpose
        flat_array = raster_data.flatten()
        lower_bound = np.quantile(flat_array, 0.001)
        upper_bound = np.quantile(flat_array, 0.999)
        mask = (raster_data >= lower_bound) & (raster_data <= upper_bound)
        filtered_array = np.where(mask, raster_data, 0)
        min_val = filtered_array.min()
        max_val = filtered_array.max()
        rescaled_array = (filtered_array - min_val) / (max_val - min_val) * 255

        show(rescaled_array.astype(np.uint8), title="Raster Data")


def download_sample_vector_data(download_dir):
    data_dir = Path(download_dir) / "departement-31"
    output_file = data_dir / "landuse.shp"
    if output_file.exists():
        return output_file

    zip_file_url = "http://opendata.lexman.net/departement-31-Haute-Garonne.zip"
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(download_dir)

    for file_name in data_dir.iterdir():
        if file_name.stem != "landuse":
            file_name.unlink()

    assert output_file.exists()
    return output_file
