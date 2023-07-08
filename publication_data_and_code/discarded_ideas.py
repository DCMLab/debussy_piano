# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: debussy
#     language: python
#     name: debussy
# ---

# %% [markdown]
# `python3 -m pip install -U pandas plotly nbformat networkx`
# # Extract, Transform, Load
#
# Before you can run this notebook, make sure you have Python 3.10 installed and execute `pip install -r requirements.txt`.

# %% jupyter={"outputs_hidden": false}
# %reload_ext autoreload
# %autoreload 2
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wavescapes

import etl
import utils

# %% [markdown]
# Configuring the notebook to produce the defaults from the paper. For more information on available normalization methods (variable `how` below) see the section "Loading magnitude-phase matrices" below.

# %% jupyter={"outputs_hidden": false}
DEBUSSY_REPO = '..'
DATA_FOLDER = 'pickled_magnitude_phase_matrices'
os.makedirs(DATA_FOLDER, exist_ok=True)
DEFAULT_FIGURE_SIZE = 1000 #2286
EXAMPLE_FNAME = 'l123-08_preludes_ondine'
how = '0c'
indulge = True
norm_method = (how, indulge)

# %% [markdown]
# ## Loading metadata
# Metadata for all pieces contained in the dataset.

# %% jupyter={"outputs_hidden": false}
metadata = etl.get_metadata(DEBUSSY_REPO)
metadata.head()

# %% [markdown]
# ### Columns for ordinal plots
#
# Creating a column years_ordinal that represents the year of publication as a range of years in which Debussy composed.
#
# Also creating a column years_periods in which the years of publication are grouped into three periods.
#
# Periods:
#
# * 1880-1892
# * 1893-1912
# * 1913-1917
#
# src: the cambridge companion to Debussy (the phases years are not consistent accross all sources)
#

# %% jupyter={"outputs_hidden": false}
years_ordinal = {val:idx for idx, val in enumerate(np.sort(metadata.year.unique()))}
metadata['years_ordinal'] = metadata.year.apply(lambda x: years_ordinal[x])

# %% jupyter={"outputs_hidden": false}
years_periods = {}

for idx, val in enumerate(np.sort(metadata.year.unique())):
    if val < 1893:
        years_periods[val] = 0
    elif val < 1913:
        years_periods[val] = 1
    else:
        years_periods[val] = 2

metadata['years_periods'] = metadata.year.fillna(1880.0).apply(lambda x: years_periods[x])
metadata.years_ordinal.head(1),metadata.years_periods.head(1) 

# %% [markdown]
# ## Loading Pitch Class Vectors (PCVs)
# An `{fname -> pd.DataFrame}` dictionary where each `(NX12)` DataFrame contains the absolute durations (expressed in quarter nots) of the 12 chromatic pitch classes for the `N` slices of length = 1 quarter note that make up the piece `fname`. The IntervalIndex reflects each slice's position in the piece. Set `pandas` to False to retrieve NumPy arrays without the IntervalIndex and column names.

# %% jupyter={"outputs_hidden": false}
pcvs = etl.get_pcvs(DEBUSSY_REPO, pandas=True)
etl.test_dict_keys(pcvs, metadata)
pcvs[EXAMPLE_FNAME].head(5)

# %% [markdown]
# ## Loading Pitch Class Matrices
# An `{fname -> np.array}` dictionary where each `(NxNx12)` array contains the aggregated PCVs for all segments that make up a piece. The square matrices contain values only in the upper right triangle, with the lower left beneath the diagonal is filled with zeros. The values are arranged such that row 0 correponds to the original PCV, row 1 the aggregated PCVs for all segments of length = 2 quarter notes, etc. For getting the segment reaching from slice 3 to 5 (including), i.e. length 3, the coordinates are `(2, 5)` (think x = 'length - 1' and y = index of the last slice included).
#
# The following example shows the upper left 3x3 submatrix where
# * the first three entries (which are PCVs of size 12) correspond to the pitch class distributions of the piece's first three quarternote slices,
# * the two last vectors of the second row each correspond to a sum of two adjacent vectors above, and
# * the last entry of the the third row corresponds to the sum all three PCVs.

# %%
pcms = etl.get_pcms(DEBUSSY_REPO)
etl.test_dict_keys(pcms, metadata)
print(f"Shape of the PCM for {EXAMPLE_FNAME}: {pcms[EXAMPLE_FNAME].shape}")
pcms[EXAMPLE_FNAME][:3, :3]

# %% [markdown]
# ## Loading Discrete Fourier Transforms
# `{fname -> np.array}` containing `(NxNx7)` complex matrices. For instance, here's the first element, a size 7 complex vector with DFT coefficients 0 through 6:

# %% jupyter={"outputs_hidden": false}
dfts = etl.get_dfts(DEBUSSY_REPO)
etl.test_dict_keys(dfts, metadata)
print(f"Shape of the DFT for {EXAMPLE_FNAME}: {dfts[EXAMPLE_FNAME].shape}")
dfts[EXAMPLE_FNAME][0,0]

# %% [markdown]
# You can view the 7 complex numbers as magnitude-phase pairs. In the following we use magnitude-phase-matrices of this format.

# %% jupyter={"outputs_hidden": false}
utils.get_coeff(dfts[EXAMPLE_FNAME], 0, 0, deg=True)

# %% [markdown]
# For convenience, values can also be inspected as strings where the numbers are rounded and angles are shown in degrees:

# %% jupyter={"outputs_hidden": false}
utils.get_coeff(dfts[EXAMPLE_FNAME], 0, 0, deg=True)

# %% [markdown]
# ## Loading magnitude-phase matrices
# `{fname -> np.array}` where each of the `(NxNx6x2)` matrices contains the 6 relevant DFT coefficients converted into magnitude-phase pairs where the magnitudes have undergone at least one normalization, i.e. are all within [0,1]. The first time the notebook runs, the matrices are computed and pickled to disk, from where they can be loaded on later runs.
#
# The parameter `norm_params` can be one or several `(how, indulge)` pairs where `indulge` is a boolean and `how ∈ {'0c', 'post_norm', 'max_weighted', 'max'}`.
#
# ### Normalizing magnitudes
#
# The available normalization methods for `how` are:
# * **'0c'** default normalisation, will normalise each magnitude by the 0th coefficient (which corresponds to the sum of the weight of each pitch class). This ensures onlypitch class distribution whose periodicity exactly match the coefficient's periodicity can reach the value of 1.
# * **'post_norm'** based on the 0c normalisation but "boost" the space of all normalized magnitude so the maximum magnitude observable is set to the max opacity value. This means that if any PCV in the utm given as input reaches the 0c normalized magnitude of 1, this parameter acts like the '0c' one. This magn_strat should be used with audio input mainly, as seldom PCV derived from audio data can reach the maximal value of normalized magnitude for any coefficient.
# * **'max'** set the grayscal value 1 to the maximum possible magnitude in the wavescape, and interpolate linearly all other values of magnitude based on that maximum value set to 1. Warning: will bias the visual representation in a way that the top of the visualisation will display much more magnitude than lower levels.
# * **'max_weighted'** same principle as max, except the maximum magnitude is now taken at the hierarchical level, in other words, each level will have a different opacity mapping, with the value 1 set to the maximum magnitude t this level. This normalisation is an attempt to remove the bias toward higher hierarchical level that is introduced by the 'max' magnitude process cited previously.
#
# `indulge` is an additional normalization that we apply to the magnitude based on the phase. Since magnitudes of 1 are possible only for a prototypical phase sitting on the unit circle, you can set this parameter to True to normalize the magnitudes by the maximally achievable magnitude given the phase which is bounded by straight lines between adjacent prototypes. (Musical prototypes are visualized in the [midiVERTO webApp](https://dcmlab.github.io/midiVERTO/#/analysis)) The pitch class vectors that benefit most from this normalization in terms of magnitude gain are those whose phase is exactly between two prototypes, such as the "octatonic" combination O₀,₁. The maximal "boosting" factors for the first 5 coefficients are `{1: 1.035276, 2: 1.15470, 3: 1.30656, 4: 2.0, 5: 1.035276}`. The sixth coefficient's phase can only be 0 or pi so it remains unchanged. Use this option if you want to compensate for the smaller magnitude space of the middle coefficients.

# %% jupyter={"outputs_hidden": false}
mag_phase_mx_dict = etl.get_magnitude_phase_matrices(dfts=dfts, data_folder=DATA_FOLDER, norm_params=norm_method)
etl.test_dict_keys(mag_phase_mx_dict, metadata)
print(f"Shape of the magnitude-phase matrix for {EXAMPLE_FNAME}: {mag_phase_mx_dict[EXAMPLE_FNAME].shape}")

# %% [markdown]
# ## Summary wavescapes
#
# This cell depends on the previously loaded magnitude-phase matrices, i.e. a conscious choice of a normalization method has been made above.
#
# `get_most_resonant` returns three `{fname -> nd.array}` dictionaries where for each piece, the three `(NxN)` matrices correspond to
#
# 1. the index between 0 and 5 of the most resonant of the six DFT coefficient 1 through 6
# 2. its magnitude
# 3. the inverse entropy of the 6 magnitudes
#
# The following example shows these 3 values for the bottom row of the example summary wavescape.

# %% jupyter={"outputs_hidden": false}
max_coeffs, max_mags, inv_entropies = etl.get_most_resonant(mag_phase_mx_dict)
np.column_stack((max_coeffs[EXAMPLE_FNAME][:3],
max_mags[EXAMPLE_FNAME][:3],
inv_entropies[EXAMPLE_FNAME][:3]))

# %% [markdown]
# ## Loading major, minor, and tritone correlations
#
# This cell loads pickled matrices. To re-compute correlations from pitch-class matrices, use `get_maj_min_coeffs()` for major and minor correlations and `get_ttms()` for tritone-ness matrices.

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
LONG_FORMAT = False

correl_dict = etl.get_correlations(DEBUSSY_REPO, DATA_FOLDER, long=LONG_FORMAT)
etl.test_dict_keys(correl_dict, metadata)
correl_dict[EXAMPLE_FNAME].shape

# %% [markdown]
# ## Loading pickled 9-fold vectors
#
# The function is a shortcut for
# * loading a particular kind of pickled normalized magnitude-phase-matrices
# * loading pickled tritone, major, and minor coefficients
# * concatenating them toegther

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
norm_params = ('0c', True)
ninefold_dict = etl.make_feature_vectors(DATA_FOLDER, norm_params=norm_params, long=LONG_FORMAT)
etl.test_dict_keys(ninefold_dict, metadata)
ninefold_dict[EXAMPLE_FNAME].shape

# %% [markdown]
# # Metrics
#
# In this section, a dataframe containing all metrics is compiled. Optional plots and tests can be done by adjusting the parameters of the wrapper function `get_metric` that can be found in `etl.py`. 

# %%
metadata_metrics = metadata.copy()
#metadata_metrics = pd.read_csv('metrics.csv').set_index('fname')


# %% [markdown]
# ## Center of mass
#
# Computing the center of mass of each coefficient for all the pieces. Uses `mag_phase_mx_dict` as input and outputs the vertical center of mass as a fraction of the height of the wavescape.

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
cols = [f"center_of_mass_{i}" for i in range(1,7)]
metadata_metrics = etl.get_metric('center_of_mass', metadata_metrics, 
                              mag_phase_mx_dict=mag_phase_mx_dict, 
                              cols=cols, store_matrix=True, 
                              show_plot=True, save_name='center_of_mass', title='Center of Mass')
metadata_metrics.head(1)

# %%
# trying out some options of the function
# 1 unified plot
metadata_metrics = etl.get_metric('center_of_mass', metadata_metrics, 
                              mag_phase_mx_dict=mag_phase_mx_dict,
                              cols=cols, store_matrix=True, 
                              show_plot=True, save_name='center_of_mass', title='Center of Mass',
                              unified=True)


# %%
# 2 using ordinal years
metadata_metrics = etl.get_metric('center_of_mass', metadata_metrics,
                              mag_phase_mx_dict=mag_phase_mx_dict,
                              cols=cols, store_matrix=True,
                              show_plot=True, save_name='center_of_mass', title='Center of Mass',
                              ordinal=True, ordinal_col='years_ordinal')

# %%
# boxplot version with ordinal column
metadata_metrics = etl.get_metric('center_of_mass', metadata_metrics, 
                              mag_phase_mx_dict=mag_phase_mx_dict,
                              cols=cols, store_matrix=True, 
                              show_plot=True, save_name='center_of_mass', title='Center of Mass', 
                              boxplot=True, ordinal=True, ordinal_col='years_periods')

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# 4. testing option
metadata_metrics = etl.get_metric('center_of_mass', metadata_metrics,
                              mag_phase_mx_dict=mag_phase_mx_dict, cols=cols,
                              store_matrix=True, testing=True)

# %% [markdown]
# # Mean Resonance
#
# Computing the mean resonance of each coefficient for all the pieces. Uses `mag_phase_mx_dict` as input and outputs the magnitude resonance of the wavescape for each coefficient.

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
cols = [f"mean_resonances_{i}" for i in range(1,7)]

metadata_metrics = etl.get_metric('mean_resonance', metadata_metrics, 
                              mag_phase_mx_dict=mag_phase_mx_dict,
                              cols=cols, store_matrix=True, 
                              show_plot=True, save_name='mean_resonances', title='Mean Resonance')
metadata_metrics.head(1)

# %%
# per period ordinal plot
metadata_metrics = etl.get_metric('mean_resonance', metadata_metrics, 
                              mag_phase_mx_dict=mag_phase_mx_dict,
                              cols=cols, store_matrix=True, 
                              show_plot=False, testing=True, save_name='mean_resonance_per_period', title='Mean Resonance per Period', boxplot=True,
                              ordinal=True, ordinal_col='years_periods')

# %% [markdown]
# # Moment of Inertia
#
# Moment of inertia of coefficient $n$ in the summary wavescape: $I(n)=1/N \sum_{i \in S(n)} w_i y_i^2$, where N is the total number of nodes in the wavescape, $S(n)$ is the set of the indices of the nodes in the summary wavescapes that are attributed to coefficient $n$ (i.e., where coefficient n is the most prominent among the six), $w_i$ is the weight (opacity) of the $i$-th node in the summary wavescape, and $y_i$ is the vertical coordinate of the $i$-th node in the summary wavescape
#

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
cols = [f"moments_of_inertia_{i}" for i in range(1,7)]
print(len(cols))
metadata_metrics = etl.get_metric('moment_of_inertia', metadata_metrics, 
                              max_coeffs=max_coeffs,
                              max_mags=max_mags,
                              cols=cols, store_matrix=True,
                              testing=True, 
                              show_plot=True, save_name='moments_of_inertia', title='Moments of Inertia', unified=True)
metadata_metrics.head(1)

# %% [markdown]
# # Prevalence of each coefficient
#
# Prevalence of coefficient $n$ in a piece: $W(n)=1/N \sum_{i \in S(n)} i$ where $N$ is the total number of nodes in the wavescape, $S(n)$ is the set of the indices of the nodes in the summary wavescapes that are attributed to coefficient $n$ (i.e., where coefficient $n$ is the most prominent among the six).

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
cols = [f"percentage_resonances_{i}" for i in range(1,7)]

metadata_metrics = etl.get_metric('percentage_resonance', metadata_metrics, 
                              max_coeffs=max_coeffs,
                              cols=cols, store_matrix=True, testing=True,
                              show_plot=True, save_name='percentage_resonance', title='Percentage Resonance', unified=True)
metadata_metrics.head(1)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# metadata_metrics = get_metric('percentage_resonance', metadata_metrics, 
#                               max_coeffs=max_coeffs,
#                               cols=cols, store_matrix=True, 
#                               show_plot=True, save_name='percentage_resonance_periods', title='Percentage Resonance (Periods)',  boxplot=True,
#                               ordinal=True, ordinal_col='years_periods')

# %% [markdown]
# In order to account for the certainty that a certain coefficient is actually the most resonance, we weigh the previous metric by entropy as follows: $W(n)=1/N \sum_{i \in S(n)} w_i$ where $N$ is the total number of nodes in the wavescape, $S(n)$ is the set of the indices of the nodes in the summary wavescapes that are attributed to coefficient $n$ (i.e., where coefficient $n$ is the most prominent among the six), and $w_i$ is the weight (opacity) of the $i$-th node in the summary wavescape, in this case, the entropy of $i$.
#
#

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
cols = [f"percentage_resonances_entropy_{i}" for i in range(1,7)]

metadata_metrics = etl.get_metric('percentage_resonance_entropy', metadata_metrics, 
                              max_coeffs=max_coeffs,
                              inv_entropies=inv_entropies,
                              cols=cols, store_matrix=True, 
                              testing=True,
                              show_plot=True, save_name='percentage_resonance_entropy', title='Percentage Resonance (entropy)', unified=True)
metadata_metrics.head(1)

# %% [markdown]
# # Decreasing magnitude in height
#
# The inverse coherence is the slope of the regression line that starts from the magnitude resonance in the summary wavescape at bottom of the wavescape and reaches the one at the top of the wavescape.

# %%
cols = 'inverse_coherence'
metadata_metrics = etl.get_metric('inverse_coherence', metadata_metrics, 
                              max_mags=max_mags,
                              cols=cols, store_matrix=True, 
                              show_plot=True, save_name='inverse_coherence', title='Inverse Coherence', unified=True, scatter=True)
metadata_metrics.head(1)

# %%
metadata_metrics.head(1)

# %%
metadata_metrics.to_csv('results/results.csv')

# %%
metadata_metrics.sort_values('inverse_coherence').tail()

# %%
max_mag = max_mags[EXAMPLE_FNAME]
#max_coeff = max_coeffs['l108_morceau']
np.polyfit((max_mag.shape[1] - np.arange(max_mag.shape[1]))/max_mag.shape[1], np.mean(max_mag, axis=0), 1)[0]

# %%
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.regplot(x=(max_mag.shape[1] - np.arange(max_mag.shape[1]))/max_mag.shape[1], y=np.mean(max_mag, axis=0), ci=False)
ax.set_title('Regression line. Example: Ondine')
ax.set_xlabel('hierarchical height')
ax.set_ylabel('mean maximum magnitude')

plt.tight_layout()
plt.savefig('figures/coherence.png')

plt.show()


# %%
metadata_metrics = etl.get_metric('inverse_coherence', metadata_metrics, 
                              max_mags=max_mags,
                              cols=cols, store_matrix=True, 
                              show_plot=False, testing=True)

# %% [markdown]
# Storing the final metrics for future use:

# %%
metadata_metrics.reset_index().to_csv('normalized_coherence.csv')

# %% jupyter={"outputs_hidden": false} pycharm={"is_executing": true, "name": "#%%\n"}
metadata_metrics.reset_index().to_csv('metrics_new.csv')

# %%
