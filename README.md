# Edge-centric time series analysis
Code to reproduce the analysis and figures in:

Leonardo Novelli and Adeel Razi (2021).
[A mathematical perspective on edge-centric functional connectivity](https://arxiv.org/abs/2106.10631).
arXiv:2106.10631.

# System requirements and instructions for use
The MATLAB code has been tested using MATLAB(R2020b) on both Linux and Windows. To run the demo on the small simulated dataset provided, just clone or unzip the repository and run `main.m`. Restricting the analysis to 10 parcels (default) will only take a few seconds on a standard laptop, while analysing all 200 parcels will take up to a few hours and 10GB of RAM. The main script will iteratively call `efc_single_subject.m` and finally call `plots.m` to generate the figures summarising the analysis results. To run the code on your own dataset, simply replace `example_data.mat` with your own 3D array (n_regions x time_steps x subjects).

# License
GPL-3.0