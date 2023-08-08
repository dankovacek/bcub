Automated Basin Delineation and Attribute Extraction
====================================================

This repository provides an automated pipeline for generating large
samples of basins from DEM. First, DEM files are collected from an
open-source repository, then assembled into a raster tile mosaic. The
study region (British Columbia) is broken into “complete” basin
sub-regions, defined by boundaries crossed only by outflowing stream
networks. The polygons describing these sub-regions are used to create
clipped rasters, which are then hydraulically enforced (fill depressions
and resolve flats) to create flow direction, flow accumulation, and
stream network rasters. The stream network raster is used as a binary
mask to identify river confluences to use for the final step, basin
delineation.

The resulting collection of basins is an estimate of a decision space
for the network optimization problem. The basin delineation process
raises interesting questions about digital representation of stream
networks and basin attributes. In the DEM preprocessing steps, we use
topography to identify stream networks. Stream networks are represented
by cells that meet a minimum flow accumulation threshold. There is no
single number to represent this minimum threshold, but here we assume a
constant value and the user should interpret these smallest headwater
basins with caution.

Set up Computing Environment
----------------------------

**Note, this code was tested on Ubuntu Linux with Python 3.10.**  
Update packages:  
&gt;`$ sudo apt update`

Install dependencies:  
&gt;`$ sudo apt-get install gdal-bin`

Clone the repository (from the root directory):  
&gt;`$ git clone https://github.com/dankovacek/bcub`

Change directories to the `bcub` folder:  
&gt;`$ cd basin_generator`

### Create virtual environment and activate to install libraries

Install pip:  
&gt;`$ sudo apt install python3-pip`

Create virtual environment at the project root level directory:  
&gt;`$ python3 -m venv env/`

Activate the virual environment:  
&gt;`$ source env/bin/activate`

Install Python packages:  
&gt;`$ pip install -r requirements.txt`

### High Performance Array Computing

The basin delineation and attribute extraction steps take a lot of time.
To speed up the process, namely for array computations on large basins,
the Jax library is used to enable GPU computation. See [installation
details](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier)
for more information. Alternatively, you can use the equivalent numpy
functions where the Jax library is used.

To get your system’s version of CUDA (on Linux): &gt;`$ nvidia-smi`

![Where to find the CUDA version for Jax library
installation.](../img/cuda_version.png)

Data Acquisition and Processing
-------------------------------

### USGS 3DEP DEM (U.S. Geological Survey 2020)

Basin polygons and terrain attributes are derived from the [USGS 3D
Elevation Program](https://www.usgs.gov/3d-elevation-program). The
product used for this dataset is the 1 arcsecond (small gaps along the
Alaska-Yukon border are infilled with 2 arcsecond data). The tiles can
be downloaded from the [USGS map
downloader](https://apps.nationalmap.gov/downloader/). A text file
pre-populated with the links to covering tiles is provided in this repo.
The tiles can be downloaded and merged into a virtual raster with gdal
by running the `get_3DEP_DEM.py` script saved under `setup_scripts`:  
&gt;`$ python get_3DEP_DEM.py`

> **Warning**<br> **The tile list urls will at some point change**:
> After downloading, compare the study region polygon with the tile set
> (vrt) to ensure al covering tiles are downloaded. Links to invidivual
> DEM tiles look like the following:  
> `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/historical/n62w130/USGS_1_n62w130_20130911.tif`

The .vrt mosaic created by the script will look similar to the image at
left when viewed in QGIS or similar software. After clipping rasters and
reassembling the resulting files to another `.vrt`, it should look like
the image at right:

![DEM Mosaic of BC and administrative boundary
regions](../img/DEM_tiled_trimmed.png)

Sub-region Polygons
-------------------

The study region is split into sub-regions that describe “complete
basins”, in other words the region bounds have no inflows, only
outflows. This is an important property when delineating basins at
arbitrary points in space. The sub-regions are derived from Water Survey
of Canada sub-sub-catchment (SSC) polygons from the National
Hydrographic Network (NHN) and from the USGS for southeast Alaska.

A zipped archive file of region polygons is provided in
`input_data/region_polygons.zip`. Unzip the folder in its existing
location.

> `$ unzip input_data/region_polygons.zip -d input_data/region_polygons/`

![Merging process for complete sub-regions.](../img/merging_regions.png)

> :warning: **Sub-region naming does not perfectly follow the NHN WUL
> naming convention.**

DEM Processing
--------------

Here we clip DEM files using the sub-region polygons because the study
region is too large to process as a whole.
[Whiteboxtools](https://www.whiteboxgeo.com/manual/wbt_book/intro.html)
is used here for the DEM processing steps of hydraulic conditioning,
flow direction, accumulation, and stream network generation.

### Clip DEM by sub-region and process stream networks

Create the individual region DEM files using the provided region
polygons and the DEM tile mosaic created in the previous step:  
&gt;`$ cd setup_scripts/`  
&gt;`$ python clip_region_DEM.py`

> **Note**<br> **Check the list of region polygons to process**: The dem
> processing scripts are initialized to test just the smallest region
> (08P - Skagit basin in Washington / BC). To process all regions,
> comment out the line `region_codes = ['08P']`

Process the region DEMs to create rasters representing flow direction,
flow accumulation, and stream network:  
&gt;`$ python derive_flow_accumulation.py`

Generate pour points
--------------------

Using the stream raster, generate pour points at river confluences.
Confluences are defined as stream cells with more than one inflow. An
inflow is an adjacent stream cell whose flow direction points to the
focal cell. &gt;`$ python find_pour_points.py`

### Hydrographic features dataset

The last step before basin delineation is to filter spurious pour
points.

> **Note**<br> The hydrographic features file is large (14 GB, 29 GB
> uncompressed) and may take a while to download. The file is not
> included in this repository.

First, download the hydrographic features dataset
(\`rhn\_nhn\_hhyd.gpkg.zip\`\`) from the [National Hydrographic
Network](https://open.canada.ca/data/en/dataset/a4b190fe-e090-4e6d-881e-b87956c07977).

Create a folder for the NHN data (*from the root directory*):
&gt;`$ mkdir input_data/NHN_data` Specify the new directory as the
destination for the download using wget (alternatively just visit the
link and download the file manually):
&gt;`$ wget https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/geobase_nhn_rhn/gpkg_en/CA/rhn_nhn_hhyd.gpkg.zip -P input_data/NHN_data`
Unzip the file:
&gt;`$ unzip -j -d input_data/NHN_data input_data/NHN_data/rhn_nhn_hhyd.gpkg.zip`
Remove the zip file:
&gt;`$ rm input_data/NHN_data/rhn_nhn_hhyd.gpkg.zip`

> **Warning**<br> **There is currently a bug preventing geopandas/fiona
> from opening this file**: The bug seems to be related to this
> [issue](https://github.com/Toblerity/Fiona/issues/1270). A PR appears
> to have been submitted to resolve the issue, so check for updates. As
> a workaround, the `pyogrio` engine is specified and a bounding box is
> provided at import. see the `lakes_filter.py`.

### Filter spurious pour points

Once the file is downloaded, the `lakes_filter` script will clip the NHN
water bodies features to each sub-region polygon to reduce RAM usage.
The water body polygons are then used to filter out (spurious)
confluences in lakes. &gt;`$ python lakes_filter.py`

Land cover and soil data layers
-------------------------------

Land cover rasters can be downloaded from the [North American Land
Change Monitoring System
(NALCMS)](http://www.cec.org/north-american-land-change-monitoring-system/).

Download the files you want to work with (2010, 2015, and 2020 land
cover rasters are available) from the link above, and keep note of the
file path where the files are saved or save them to a new folder at
`input_data/NALCMS/`. The files are large and may take a while to
download. The files are not included in this repository.

The soil permeability and porosity information is contained in the
[GLobal HYdrogeology MaPS
(GLHYMPS)](https://borealisdata.ca/dataset.xhtml?persistentId=doi%3A10.5683/SP2/TTJNIU)
dataset. Download the files you want to work with from the link above,
and keep note of the file path where the files are saved or save them to
a new folder at `input_data/GLHYMPS/`. The file is large and may take a
while to download. The files are not included in this repository.

The GLHYMPS dataset contains global coverage, so you may first want to
clip it to the bounding box of the study region. The bounding box is
provided in the `input_data/` folder. The bounding box is a shapefile,
so you can use QGIS or similar software to clip the GLHYMPS dataset and
reproject. The clipped file should be saved to the `input_data/GLHYMPS/`
folder.

### Basin delineation

The data preparation work is nearly complete, now we generate a large
sample of basins to characterize the decision space (of candidate
monitoring locations).

Generate a basin for each of the pour points:  
&gt;`$ setup_scripts/python derive_basins.py`

This script will output a file in parquet format which is a compressed,
columnar data format. To save in geojson format to read in QGIS, comment
out the `to_parquet()` line and uncomment the next line
`merged_basins.to_file(output_fpath.replace('.parquet', '.geojson'), driver='GeoJSON')`.
Note that the file will be very large because of the polygon geometry.
The parquet format is more efficient for reading and writing, but
geojson can be viewed in QGIS.

Basin Attribute Extraction
--------------------------

Two methods are provided to extract attributes from the basins. The
first, `extract_attributes.py`, extracts attributes from the basins and
saves them to a `geojson` file. The second, `populate_postgis.py`, first
creates a Postgres database with the PostGIS extension and then
populates the database. The first method is more direct and easier to
use, but the second method yields a format that is more performant for
spatial querying.

### Direct Extraction

The `extract_attributes_direct.py` script will extract attributes from
the basin polygon sets and save them to a `geojson` file. The attributes
are extracted from the DEM, land cover, and soil data layers. Since the
pour points are unique, the pour point geometry column is used to index
rows between the basin geometry (polygon) files and the attribute files.
This makes interacting with the data more performant since we drop the
polygon geometry column that’s responsible for most of the memory usage.

Postgres & PostGIS
------------------

### Steps to recreate:

1.  create database
2.  create schema
3.  create table
4.  populate table

The `create_database.py` script assumes Postgres is installed, and a
user (with password) and a database have been created. See additional
notes below for a few useful commands.

Towards the bottom of `create_database.py`, there are four variables
that are required to establish a database connection. `db_host`,
`db_name`, `db_user`, and `db_password`. Update these variables to match
your database configuration.

For a details on setting up Postgres and PostGIS, see [this
tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-22-04).

> **Note**<br> `db_host` is typically localhost, but if you are
> connecting to a remote database, you will need to update this
> variable.

### Create a Database

Create a database named `basins`:
&gt;`$ sudo -u postgres createdb basins`

#### Enable PostGIS

Log into the database as the superuser (postgres) and enable the PostGIS
extension: &gt;`$ sudo -u postgres psql`

Switch to the ‘basins’ database &gt;`postgres=# \c basins`

Enable the PostGIS extension: &gt;`postgres=# CREATE EXTENSION postgis;`

Restart the database service after any configuration change:
&gt;`$ sudo systemctl restart postgresql`

### Create database tables for basin geometry

Run the `create_database.py` script to create the tables and populate
the database. The schema should be automatically created from the
parquet file created from executing the `derive_basins.py` script. The
`create_database` script will create the attributes table in the
`basins` database you created and populate it with geometry and basic
metadata.

### Extend the Database

The `extend_postgis_database.py` script will add the remaining
attributes. The `extend_postgis_database.py` script will take a while to
run, so it is recommended to run it in a `tmux` or `screen` session.
With PostGIS we can create raster tables which can be then used in
conjunction with the polygons to derive the same set of attributes as
generated in using the direct attribute extraction method.

Additional Notes
----------------

### List columns in a database table

Once the `create_database` script has been executed successfully:
&gt;`\d basins_schema.basin_attributes`

Create raster table
-------------------

> raster2pgsql -s 3005 -f nalcms\_2010\_raster -I -C -M
> Clipped\_NALCMS\_2010\_land\_cover\_30m\_3005.tif -t 97x175
> basins\_schema.nalcms\_2010 | psql -d basins

Repeat for other geospatial layers.

### PostgreSQL basics

[Instructions from
DigitalOcean](https://www.digitalocean.com/community/tutorials/how-to-install-postgresql-on-ubuntu-22-04-quickstart).

Switch to postgres role: `sudo -i -u postgres`

Access postgres prompt: `psql`

Change databases: `\c <db_name>`

Quit postgres prompt: `\q`

Return to regular system: `exit`

Change user: `sudo -i -u postgres`

basins database info:

`user: postgres` `pass: <your_password>`

Create a new database

`postgres@server:~$ createdb basins`

Create the postgis extension (done from the psql terminal after
connecting to “basins” db): &gt;`CREATE EXTENSION postgis;`

<!-- Automate citation formatting for the README document.

>`pandoc -t markdown_strict -citeproc README-draft.md -o README.md --bibliography bib/bibliography.bib` -->

Via the RichDEM documentation, the aspect of a basin is the direction of
the maximum slope of the focal cell (in degrees), from Hill (1981).

Next Steps
----------

U.S. Geological Survey. 2020. “USGS 3D Elevation Program Digital
Elevation Model.”
<https://data.usgs.gov/datacatalog/data/USGS:35f9c4d4-b113-4c8d-8691-47c428c29a5b>.
