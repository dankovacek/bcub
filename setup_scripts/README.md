Automated Basin Delineation
===========================

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
&gt;`cd basin_generator`

### Create virtual environment and activate to install libraries

Install pip:  
&gt;`sudo apt install python3-pip`

Create virtual environment at the project root level directory:  
&gt;`python3 -m venv env/`

Activate the virual environment:  
&gt;`source env/bin/activate`

Install Python packages:  
&gt;`pip install -r requirements.txt`

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
&gt;`python get_3DEP_DEM.py`

> :warning: **The tile list urls will at some point change**: After
> downloading, compare the study region polygon with the tile set (vrt)
> to ensure al covering tiles are downloaded. Links to invidivual DEM
> tiles look like the following:  
> `http://mirrors.iplantcollaborative.org/earthenv_dem_data/EarthEnv-DEM90/EarthEnv-DEM90_N55W125.tar.gz`

The .vrt mosaic created by the script looks like the image at left, and
at right after clipping rasters using the provided sub-region polygons:

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

> `unzip input_data/region_polygons.zip -d input_data/region_polygons/`

![Merging process for complete sub-regions.](../img/merging_regions.png)

> :warning: **Sub-region naming may not perfectly follow the WUL naming
> convention.**

DEM Processing
--------------

Here we clip DEM files using the sub-region polygons because the study
region is too large to process as a whole.
[Whiteboxtools](https://www.whiteboxgeo.com/manual/wbt_book/intro.html)
is used here for the DEM processing steps of hydraulic conditioning,
flow direction, accumulation, and stream network generation.

Create the individual region DEM files using the provided region
polygons and the DEM tile mosaic created in the previous step:  
&gt;`cd setup_scripts/`  
&gt;`python clip_region_DEM.py`

> :wave: **Check the list of region polygons to process**: The above
> script is initialized to test just the smallest region. To process all
> regions, comment out the line `region_codes = ['08P']`

Process the region DEMs to create rasters representing flow direction,
flow accumulation, and stream network:  
&gt;`python derive_flow_accumulation.py`

Using the stream raster, generate pour points at headwaters and
confluences:  
&gt;`python automate_pourpt_generation.py`

The data preparation work is done. Now we generate large sample of
basins to characterize the decision space (of candidate monitoring
locations)

Generate a basin for each of the pour points:  
&gt;`setup_scripts/python pysheds_derive_basin_polygons.py`

We next run a Monte Carlo (MC) simulation to characterize the cumulative
probability distribution of basin attributes based on different basin
pour point selection methods. Deriving a basin for each stream cell
(candidate location) is not feasible, so we want to develop more
efficient characterization methods and compare the resulting
characterization of attributes.

The MC simulation procedure is as follows:

1.  Set the “monitoring area density”, corresponding to one station per
    *A*<sub>*s*</sub> km^2 to represent the number of locations selected
    in each simulation. A 1000 km^2 study region then has
    *N*<sub>*s**t**n*</sub> = 10 stations *in each simulation*. Note
    that the WMO guideline for monitoring networks in mountainous
    regions is one station per 1000 *k**m*<sup>2</sup> (WMO, 2008), so
    our simulations are 10x more dense than the guideline.  

2.  Set the number of simulations *N*<sub>*s**i**m*</sub>. There is a
    performance tradeoff between the number of stations per simulation
    and the number of simulations that needs to be explored to better
    optimize the characterization process.

3.  Retrieve the indices of all stream cells in the processed
    (hydraulically conditioned) raster.

4.  Generate *N*<sub>*s**t**n*</sub> pour points (candidate monitoring
    locations) from the set of stream cells by three selection methods:

    1.  **Random (RAND)**: randomly select *N*<sub>*s**t**n*</sub> cells
        from the array of all stream cells.

    2.  **Confluence (CONF)**: using the flow direction raster, find all
        confluences. Confluences are defined as stream cells fed by more
        than one connected stream cell using the D8 flow direction
        convention. Randomly select *N*<sub>*s**t**n*</sub> confluence
        cells.

    3.  **Gradient (GRAD)**: using the flow accumulation raster, find
        all cells where the difference in accumulation between connected
        cells is greater than some threshold. We are interested in
        minimizing the redundancy in basin delineation, for example a
        basin comprised of 10<sup>6</sup> cells and its immediate
        neighbour which adds only a marginal contributing basin area
        would yield two basins with nearly identical attributes. The
        threshold then represents a minimum area change we are
        interested in tracking. The threshold
        (*a*<sub>*g**r**a**d*</sub>) is set to 5% of the target cell
        accumulation.

    <!-- -->

        For all stream cells:  
            (i, j) = stream cell indices
            acc = upstream contributing area (constant scalar)  
            W = 3x3 matrix with acc at index 1, 1  
            G = W - acc  
            pct = G / acc
                if any(G) > $a_{grad}$:  
                    append (i, j) to a list.

MAKE REV D THE PERCENT-BASED ACCUMULATION GRADIENT

Additional Notes
----------------

<!-- Automate citation formatting for the README document.

>`pandoc -t markdown_strict -citeproc README-draft.md -o README.md --bibliography bib/bibliography.bib` -->

Via the RichDEM documentation, the aspect of a basin is the direction of
the maximum slope of the focal cell (in degrees), from Hill (1981).

Next Steps
----------

Attribute Extraction

U.S. Geological Survey. 2020. “USGS 3D Elevation Program Digital
Elevation Model.”
<https://data.usgs.gov/datacatalog/data/USGS:35f9c4d4-b113-4c8d-8691-47c428c29a5b>.
