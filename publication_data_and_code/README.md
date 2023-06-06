<!-- TOC -->
* [Code for reproducing data, results, and figures](#code-for-reproducing-data-results-and-figures)
  * [Running the Python code](#running-the-python-code)
    * [Installing requirements](#installing-requirements)
    * [Running the code](#running-the-code)
  * [Running the R code](#running-the-r-code)
* [Re-generating auxiliary files](#re-generating-auxiliary-files)
  * [Re-generating the file with Pitch Class Vectors (PCVs) for all pieces](#re-generating-the-file-with-pitch-class-vectors-pcvs-for-all-pieces)
  * [Re-generating (and inspecting) median durations of matching Spotify recordings](#re-generating-and-inspecting-median-durations-of-matching-spotify-recordings)
* [Extra: Commandline script for creating all wavescapes, including other normalization methods](#extra-commandline-script-for-creating-all-wavescapes-including-other-normalization-methods)
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

# Re-generating auxiliary files

## Re-generating the file with Pitch Class Vectors (PCVs) for all pieces

If `dimcat==0.3.0` is installed (part of the requirements.txt), you can head to your clone of this repository and run:

```bash
dimcat pcvs -q 1.0 -p pc -w 0.5 --fillna 0.0 --round 5
```

This will generate the file `all-1.0q_sliced-w0.5-pc-pcvs.tsv` containing all quarter-note-slice PCVs of all pieces.
We renamed this file to `debussy-1.0q_sliced-w0.5-pc-pcvs.tsv` and the function `etl.get_pcvs()` is hard-coded to use 
this one.

The parameters correspond to the following configuration of PCVs (and could be adapted for further studies):

* `-q 1.0` score slices of length 1 quarter
* `-p pc` using pitch classes 0..11
* `-w 0.5` weighting grace notes by half of their durations
* `--fillna 0.0` fills empty fields (=non-occurrent pitch classes) with 0.0
* `--round 5` rounds the output to 5 (maximum available precision).

The documentation of all parameter options can be accessed via `dimcat pcvs -h`.

## Re-generating (and inspecting) median durations of matching Spotify recordings

Within the `generate_data_and_metricy.ipynb` notebook, the `concatenated_metadata.tsv` is enriched with the columns `median_recording` 
from which the tempo-like indications `qb_per_minute` (quarter beats per minute) and `sounding_notes_per_minute` are computed. 
The median recording times are stored in `durations/spotify_median_durations.json` and can be re-computed by heading into the 
`durations` folder, running `pip install -r requirements.txt` in the virtual environment that the Jupyter notebook will have access to, 
and running the notebook `spotify_durations.ipynb`. By default, it uses the cached search results in `spotify_search_results.json` but
the notebook can be configured to re-generate these, too, by providing a [Spotify API access token](https://developer.spotify.com/documentation/web-api).
The notebook also generates an interactive version of the following box plots that summarizes the durations of all matched recordings
taken into account (in seconds):

![newplot(5)](https://github.com/DCMLab/debussy_piano/assets/42718519/5d366cac-fc3e-4be2-9b58-1d007ac90f22)

By the badly labeled axes you can tell that this code is less maintained (or documented) than the rest. If you encounter any bugs or 
have questions, feel free to [leave us an issue](https://github.com/DCMLab/debussy_piano/issues).


# Extra: Commandline script for creating all wavescapes, including other normalization methods

This script is an additional gimmick and lets you create wavescape figures for all pieces and with settings of your own choice.
Technically, wavescapes are visualizations of what we call "magnitude-phase matrices" which correspond to upper-triangle matrices (UTM)
containing the six Fourier coefficients for each slice and for each possible sum of adjacent slices, with one of the eight 
possible normalization methods applied. Therefore, when you run the script, these matrices are computed and pickled to disk first.

If you have run the code from the previous section, the mag-phase matrices for the `0c+indulge` normalization have already been 
written to the folder `pickled_magnitude_phase_matrices`. They correspond to what you get by navigating to the `publication_data_and_code`
folder and executing the following command:

    python create_data.py pickled_magnitude_phase_matrices/ -n 4

(only that here you're getting 82 additional files containing the correlation of all PCV UTMs with the durational pitch-class profiles 
computed from the [Mozart Piano Sonatas](https://github.com/DCMLab/mozart_piano_sonatas/)).


Add `-w wavescape_plots` to additionally create wavescapes in the folder `wavescape_plots` (or whatever name you want to pick), too.

**Please be warned that, by default, all your CPU cores will be used in parallel to speed up the computation but that this will 
likely fill up your entire working memory which might slow down or even crash your computer. Reduce the number of cores using
the `-c` option, or even set it to -1 to use a for-loop (very slow).**

You will get 1148 plots for normalization methods 0-3 or 984 plots for normalization methods 4-7:

For each of the 82 pieces, the script will generate a greyscale and a colored wavescape for each of the 6 coefficients, plus two different
summary wavescapes, i.e. 82 * (2 * 6 + 2) = 1148. The reason why methods 4-7 yield only 82 * (2 * 5 + 2) wavescapes is that they add 
the "indulge-prototypes" normalization which does not change the wavescapes for the sixth coefficient. In the example above, the 164 
missing wavescapes can be addid via `python create_data.py pickled_magnitude_phase_matrices/ -n 0 -w wavescape_plots --coeffs 6`, 
i.e. only normalization method 0 for coefficient 6.

The documentation of all parameter options can be accessed via `python create_data.py -h`. The four normalization methods and their
`indulge_prototypes` variants are documented in the 
[`normalize_dft()` docstring](https://github.com/DCMLab/wavescapes/blob/master/wavescapes/dft.py#L202) of the 
wavescapes package.



