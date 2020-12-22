## Introduction

This repository is used to reproduce the results of the Distribution-Learning-based-weighting (DLW) method, including simulations and real data. Note that the baselines are implemented in R and the DLW method is implemented in Python.

## Requirements
The code was tested with R 3.6.1, python 3.7 and pytorch 1.7.0. Please install the dependencies for R and Python first. 

```
# install R packages
Rscript R_requirements.R

# intall Python packages
pip install -r python_requirements.txt

```
## Run the code

To reproduce the simulations, we recommend that you generate data and implement those baseline methods in R first, then run the python code to implement the DLW method and get all the results.

```
# examples to implement data generation and baseline methods in R
cd baselines_R
mkdir data_simu
mkdir R_result
Rscript simu2 8 10000 0.4 10

# examples to implement DLW method in python
python3 density_weighting.py --simu_class=simu2 --d=8 --n=10000 --sc=0.4 --times=10



