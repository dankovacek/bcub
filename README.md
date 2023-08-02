# British Columbia Ungauged Basin Dataset

## Introduction

To support research in streamflow monitoring network optimization, we need a dataset to represent the set of possible monitoring locations.  The British Columbia Ungauged Basin (BCUB) dataset is intended to represent an approximate decision space for the optimization problem.  

This repository contains the codebase used to generate the dataset.  Users may find a [condensed results file]()containing basin pour point coordinates and basin attributes, or may use any of the [basin region polygons (region_polygons.zip)](https://github.com/dankovacek/bcub/blob/main/input_data/) provided to derive the sample for some or all of the study region.  A [detailed example](https://dankovacek.github.io/bcub_demo) is provided in the form of a Jupyter book to demonstrate the full process on a smaller region, Vancouver Island.

See the [README.md under `setup_scripts/`](https://github.com/dankovacek/bcub/tree/main/setup_scripts) for setup of the validation scripting.  

## Disclaimer and Performance Notes

The code provided in this repository is not intended to be a production-ready tool.  The code is provided as-is, and is **intended to be used as a reference for users to understand the process used to generate the dataset**.  The complete dataset was generated on an Intel Xeon CPU E5-2690 @ 2.60GHz with 128GB RAM running Ubuntu Linux.  The large ram is necessary to process basins in the larger regions (Peace, Liard, Fraser).  Otherwise, the smaller regions were tested on six year old Intel i7-8850H CPU @ 2.60GHz running Ubuntu Linux with an NVIDIA Quadro P2000 GPU.  To get around the RAM issue, use lower resolution DEM, such as [EarthENV DEM90](https://www.earthenv.org/DEM), or set up the data processing in a much smarter way to aggregate smaller regions like the [Caravan dataset](https://github.com/kratzert/Caravan). 

Processing this dataset took plenty of time, and [YMMV](https://dictionary.cambridge.org/dictionary/english/ymmv).

## License

The code is provided under the [Creative Commons Attribution 4.0 International](https://github.com/dankovacek/bcub/blob/main/LICENSE) license.

## References
