import io
import time
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr
import zarr
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
        rescaled_array = rescale_raster(raster_data)

        show(rescaled_array.astype(np.uint8), title="Raster Data")


def rescale_raster(raster_data):
    flat_array = raster_data.flatten()
    lower_bound = np.quantile(flat_array, 0.001)
    upper_bound = np.quantile(flat_array, 0.999)
    mask = (raster_data >= lower_bound) & (raster_data <= upper_bound)
    filtered_array = np.where(mask, raster_data, 0)
    min_val = filtered_array.min()
    max_val = filtered_array.max()
    rescaled_array = (filtered_array - min_val) / (max_val - min_val) * 255

    return rescaled_array


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


def generate_hdf5_file(output_file, raster_file):
    # assumes raster with 3 bands
    output_file = Path(output_file)

    if not Path(raster_file).exists():
        download_sample_data(output_file.parent)

    if raster_file:
        with rasterio.open(raster_file, "r") as src:
            image_data = src.read()
            data_2d = image_data[0]  # 1st raster band
            data_3d = image_data  # all raster bands
            desc_2d = "st band from a raster"
            desc_3d = "RGB bands from a raster"
    else:
        data_2d = np.random.rand(10, 10)  # 10x10 random array
        data_3d = np.random.rand(5, 10, 10)  # 5x10x10 random array
        desc_2d = "Random 2D data"
        desc_3d = "Random 3D data"

    with h5py.File(output_file, 'w') as f:
        # Main group
        group = f.create_group("data")

        # 2D dataset (lon, lat)
        dset_2d = group.create_dataset("data_2d", data=data_2d)
        dset_2d.attrs["description"] = desc_2d
        dset_2d.dims[0].label = "latitude"
        dset_2d.dims[1].label = "longitude"

        # 3D dataset (time, lon, lat)
        dset_3d = group.create_dataset("data_3d", data=data_3d)
        dset_3d.attrs["description"] = desc_3d
        dset_3d.dims[0].label = "time"
        dset_3d.dims[1].label = "latitude"
        dset_3d.dims[2].label = "longitude"

        # metadata
        group.attrs["creation_date"] = str(datetime.today().strftime('%Y-%m-%d'))
        group.attrs["project"] = "Tutorial example project"


def download_netcdf(download_dir):
    output_file = Path(download_dir) / "precip.mon.total.v7.nc"
    if output_file.exists():
        return output_file

    zip_file_url = "https://www.kaggle.com/api/v1/datasets/download/bigironsphere/gpcc-monthly-precipitation-dataset-05x05"
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(download_dir)

    assert output_file.exists()
    return output_file


def compare_datacube_formats(comparison_directory, random_shape=None):
    """
    compare hdf5, netcdf and zarr files: create a random array of size (10, 1000, 1000) and for each format:
    1) measure the write time
    2) measure file size
    3) measure the read time
    """
    if not random_shape:
        random_shape = (100, 1000, 1000)
    Path(comparison_directory).mkdir(exist_ok=True)
    data = np.random.rand(*random_shape)

    # HDF5
    # write
    h5_start_write = time.time()
    h5_file = Path(comparison_directory)/"h5_file.h5"
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('array', data=data)
    h5_write_time = time.time() - h5_start_write

    # file size
    h5_file_size = get_file_size_in_mb(h5_file)

    # open
    h5_start_read = time.time()
    with h5py.File(h5_file, 'r') as f:
        data_hdf5 = f['array'][:]
    h5_read_time = time.time() - h5_start_read

    # NetCDF
    # write
    nc_start_read = time.time()
    nc_file = Path(comparison_directory)/"nc_file.nc"
    ds = xr.DataArray(data, dims=["time", "x", "y"])
    ds.to_netcdf(nc_file)
    nc_write_time = time.time() - nc_start_read

    # file size
    nc_file_size = get_file_size_in_mb(nc_file)

    # open
    nc_start_read = time.time()
    ds_netcdf = xr.open_dataset(nc_file)
    data_netcdf = ds_netcdf.values
    nc_read_time = time.time() - nc_start_read

    # Zarr
    # write
    zarr_start = time.time()
    zarr_file = str(Path(comparison_directory)/"zarr_file.zarr")
    zarr.save(zarr_file, data)
    zarr_write_time = time.time() - zarr_start

    # file size
    zarr_file_size = zarr.storage.DirectoryStore(zarr_file).getsize() * 1e-6

    # read
    zarr_start_read = time.time()
    zarr_data = zarr.open(zarr_file, mode='r')
    data_zarr = zarr_data[:]
    zarr_read_time = time.time() - zarr_start_read

    print(f"Write times:\n"
          f"\th5: {h5_write_time:.3f} seconds\n"
          f"\tnc: {nc_write_time:.3f} seconds\n"
          f"\tzarr: {zarr_write_time:.3f} seconds")

    print(f"Read times:\n"
          f"\th5: {h5_read_time:.3f} seconds\n"
          f"\tnc: {nc_read_time:.3f} seconds\n"
          f"\tzarr: {zarr_read_time:.3f} seconds")

    print(f"File sizes:\n"
          f"\th5: {h5_file_size:.3f} MB\n"
          f"\tnc: {nc_file_size:.3f} MB\n"
          f"\tzarr: {zarr_file_size:.3f} MB")