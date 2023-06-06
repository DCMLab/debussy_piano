<!-- TOC -->
* [Code for reproducing results and figures](#code-for-reproducing-results-and-figures)
  * [Running the Python code](#running-the-python-code)
    * [Installing requirements](#installing-requirements)
    * [Running the code](#running-the-code)
  * [Running the R code](#running-the-r-code)
  * [Re-generating the file with Pitch Class Vectors (PCVs) for all pieces](#re-generating-the-file-with-pitch-class-vectors-pcvs-for-all-pieces)
<!-- TOC -->

# Code for reproducing results and figures

The code consists of two components, namely

* the Python code for computing all measures and related figures for section **3. Methodology**.
* the R script `analyses_metrics.R` that computes the Bayesian mixed-effects models for section
  **4. Results**

Both require you to clone the dataset using the command

    git clone --recurse-submodules -j8 https://github.com/DCMLab/debussy_piano.git

## Running the Python code

The code can be run in two ways:

* as standalone script `generate_data_and_metrics.py`, or
* as interactive Jupyter notebook `generate_data_and_metrics.ipynb`.

In both cases you need to install the required Python packages first.

### Installing requirements

The code requires Python 3.10. You can check your current version by typing `python --version`.

Before installing the packages you may want to create a dedicated virtual environment, e.g. via
[virtualenv](www.virtualenv.org) or [conda](www.conda.io).

The required packages are listed in `requirements.txt` and can be installed via

    python -m pip install -r requirements.txt

### Running the code

If you want to run the standalone script (generated from the Jupyter notebook via 
[Jupytext](jupytext.readthedocs.io/)), navigate to the `publication_data_and_code` directory,
and run

    python generate_data_and_metrics.py

In order to execute the interactive notebook instead, you need to have [Jupyter](http://jupyter.org/install) installed and made sure that you have [installed a kernel](https://ipython.readthedocs.io/en/latest/install/kernel_install.html#kernels-for-different-environments) for the environment with the installed packages.


## Running the R code

The code can be executed in RStudio after loading the required libraries. In order to run the Bayesian models, it is necessary to install [`brms`](https://cran.r-project.org/web/packages/brms/readme/README.html) and [`rstan`](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started).

The code can be used to reproduce the analyses as reported in the paper (by setting `retrain_models <- FALSE`), or to train the models from scratch (`retrain_models <- TRUE`).

## Re-generating the file with Pitch Class Vectors (PCVs) for all pieces

If `dimcat==0.3.0` is installed (part of the requirements.txt), you can head to your clone of this repository and run:

```bash
dimcat pcvs -q 1.0 -p pc -w 0.5 --fillna 0.0 --round 5
```

This will generate the file `all-1.0q_sliced-w0.5-pc-pcvs.tsv` containing all quarter-note-slice PCVs of all pieces.
We renamed this file to `debussy-1.0q_sliced-w0.5-pc-pcvs.tsv` and the function `etl.get_pcvs()` is hard-coded to use 
this one.