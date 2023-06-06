<!-- TOC -->
* [Code for reproducing data, results, and figures](#code-for-reproducing-data-results-and-figures)
  * [Running the Python code](#running-the-python-code)
    * [Installing requirements](#installing-requirements)
    * [Running the code](#running-the-code)
  * [Running the R code](#running-the-r-code)
* [Generating a file with Pitch Class Vectors (PCVs) for all pieces](#generating-a-file-with-pitch-class-vectors-pcvs-for-all-pieces)
<!-- TOC -->

# Code for reproducing data, results, and figures

The code consists of two components, namely

* the Python code for computing all measures reported in section **3. Methodology** and some additional figures. 
  Its outputs are required for running
* the R script `analyses_metrics.R` that computes the Bayesian mixed-effects models for section
  **4. Results**


## Running the Python code

In order to run the code that generates the data and figures you need to install the required Python packages 
(see the next section) and to clone this repository with all included submodules using the command

    git clone --recurse-submodules -j8 https://github.com/DCMLab/debussy_piano.git

When run, the code needs to be executed from the `publication_data_and_code` directory where it resides and
in which it will produce the following folders and files:

* `pickled_magnitude_phase_matrices`: containing 82 pickled numpy matrices, one per piece, containing the results of
  applying the Discrete Fourier Transform to the pitch class vectors under the `0c+indulge` normalization (see section 
  `Methodology.Wavescapes` in the paper).
* `results`: containing 
  * `results.csv`, a copy of the `../concatenated_metadata.tsv` with additional columns that contain the computed 
    metrics for each piece (82 rows); and 
  * `results_melted.csv`, the same results in onther formats: the six values for each metric (one per Fourier coefficient) 
    are reproduced in the same column for analysis in R, resulting in a long-format dataframe with 6 * 82 = 492 rows.
* `figures`: Some additional figures showing the behaviour of the metrics.

### Installing requirements

The code requires Python 3.10. You can check your current version by typing `python --version`.

Before installing the packages you may want to create a dedicated virtual environment, e.g. via
[virtualenv](www.virtualenv.org) or [conda](www.conda.io).

The required packages are listed in `requirements.txt` and can be installed via

    python -m pip install -r requirements.txt

### Running the code

The code is available in two formats:

* as interactive Jupyter notebook `generate_data_and_metrics.ipynb`, or
* as standalone script `generate_data_and_metrics.py` (synchronized with the Jupyter notebook via 
  [Jupytext](jupytext.readthedocs.io/))

To run the standalone script , navigate to the `publication_data_and_code` directory, make sure to have activated your
virtual environment in which you have installed the required packages, and then run

    python generate_data_and_metrics.py

In order to execute the interactive notebook instead, you need to have [Jupyter](http://jupyter.org/install) installed and made sure that you have 
[installed a kernel](https://ipython.readthedocs.io/en/latest/install/kernel_install.html#kernels-for-different-environments) 
for the environment with the installed packages.


## Running the R code

The code can be executed in RStudio after loading the required libraries. In order to run the Bayesian models, it is necessary to install [`brms`](https://cran.r-project.org/web/packages/brms/readme/README.html) and [`rstan`](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started).

The code can be used to reproduce the analyses as reported in the paper (by setting `retrain_models <- FALSE`), or to train the models from scratch (`retrain_models <- TRUE`).

# Generating a file with Pitch Class Vectors (PCVs) for all pieces

`dimcat pcvs -q 1.0 -p pc -w 0.5 --fillna 0.0 --round 5`

