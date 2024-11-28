# Tutorial: Geospatial Data Formats

Geospatial data allows the communication of information in a wide range of fields. Over time,
many data formats have emerged to support this diversity. This tutorial introduces these various
formats, their evolution, their advantages and disadvantages, and provides guidance on how and in
which contexts to use them most effectively. It will be divided into several jupyter notebooks:

1) [Raster formats](./raster_formats.ipynb)
2) [Vector formats](./vector_data_formats.ipynb)
3) [Data cube formats](./datacube_formats.ipynb)
4) [Point clouds](./point_clouds.ipynb)
5) [Data optimization](./data_optimization.ipynb)

For each tutorial, sample data will be provided in the `sample_data` directory or automatically
downloaded (in the `sample_data` directory by default)


To run a notebook, first create a virtual environment using pip. Then, install the required 
packages (if not already done), using the `requirements.txt` file.

```bash
python -m venv venv
pip install -r requirements.txt
source venv/bin/activate
```

Then you can simply use `jupyter notebook xxx.ipynb` to run the corresponding notebook.
