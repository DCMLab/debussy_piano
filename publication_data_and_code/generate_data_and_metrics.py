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
DATA_FOLDER = '.'
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
# The column `year` contains composition years as the middle between beginning and end  of the composition span.

# %% jupyter={"outputs_hidden": false}
metadata.year.head(10)

# %% [markdown]
# Series `median_recording` contains median recording times in seconds, retrieved from the Spotify API. the Spotify API.

# %% jupyter={"outputs_hidden": false}
metadata.median_recording.head(10)

# %% [markdown]
# Columns mirroring a piece's activity are currently:
# * `qb_per_minute`: the pieces' lengths (expressed as 'qb' = quarterbeats) normalized by the median recording times; a proxy for the tempo
# * `sounding_notes_per_minute`: the summed length of all notes normalized by the piece's duration (in minutes)
# * `sounding_notes_per_qb`: the summed length of all notes normalized by the piece's length (in qb)
# Other measures of activity could be, for example, 'onsets per beat/second' or 'distinct pitch classes per beat/second'.

# %% [markdown]
# ## Loading Pitch Class Vectors (PCVs)
# An `{fname -> pd.DataFrame}` dictionary where each `(NX12)` DataFrame contains the absolute durations (expressed in quarter nots) of the 12 chromatic pitch classes for the `N` slices of length = 1 quarter note that make up the piece `fname`. The IntervalIndex reflects each slice's position in the piece. Set `pandas` to False to retrieve NumPy arrays without the IntervalIndex and column names.

# %% jupyter={"outputs_hidden": false}
pcvs = etl.get_pcvs(DEBUSSY_REPO, pandas=True)
etl.test_dict_keys(pcvs, metadata)
pcvs[EXAMPLE_FNAME].head(5)

# %% [markdown]
# The `wavescapes` library allows for creating a wavescape directly from a PCV matrix. Here is one showing the second coefficient: 

# %% jupyter={"outputs_hidden": false}
coeff = 2
label = etl.make_wavescape_label(EXAMPLE_FNAME, how, indulge, coeff=coeff)
os.makedirs('figures', exist_ok=True)
path = os.path.join('figures', etl.make_filename(EXAMPLE_FNAME, how, indulge, coeff=coeff, ext='.png'))
wavescapes.single_wavescape_from_pcvs(pcvs[EXAMPLE_FNAME],
                                      width=DEFAULT_FIGURE_SIZE,
                                      coefficient=coeff,
                                      save_label=path,
                                      magn_stra=how,
                                      output_rgba=False,
                                      ignore_phase=True,
                                      aw_per_tick=10,
                                      tick_factor=10,
                                      label=label,
                                      label_size=15)

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

# %%
dfts = etl.get_dfts(DEBUSSY_REPO)
etl.test_dict_keys(dfts, metadata)
print(f"Shape of the DFT for {EXAMPLE_FNAME}: {dfts[EXAMPLE_FNAME].shape}")
dfts[EXAMPLE_FNAME][0,0]

# %% [markdown]
# You can view the 7 complex numbers as magnitude-phase pairs. In the following we use magnitude-phase-matrices of this format.

# %%
utils.get_coeff(dfts[EXAMPLE_FNAME], 0, 0, deg=True)

# %% [markdown]
# For convenience, values can also be inspected as strings where the numbers are rounded and angles are shown in degrees:

# %%
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

# %%
mag_phase_path = os.path.join(DATA_FOLDER, 'pickled_magnitude_phase_matrices') 
os.makedirs(mag_phase_path, exist_ok=True)
mag_phase_mx_dict = etl.get_magnitude_phase_matrices(dfts=dfts, data_folder=mag_phase_path, norm_params=norm_method)
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
# In order to plot these values in a wavescape, they need to be transformed into color values. The function `most_resonant2color()` attributes six equidistant hues to the most resonant coefficients and takes one of the other two matrices to adapt the opacity.
#
# In the first example the opacity shows the magnitude of the most resonant coefficient:

# %%
colors = utils.most_resonant2color(max_coeffs[EXAMPLE_FNAME], max_mags[EXAMPLE_FNAME])
by_entropy = False
label = etl.make_wavescape_label(EXAMPLE_FNAME, how, indulge, by_entropy=by_entropy)
ws = wavescapes.Wavescape(colors, width=DEFAULT_FIGURE_SIZE)
ws.draw(aw_per_tick=10, tick_factor=10, label=label, label_size=15)#, subparts_highlighted=[110,134])
path = os.path.join('figures', etl.make_filename(EXAMPLE_FNAME, how, indulge, summary_by_entropy=by_entropy, ext='.png'))
plt.savefig(path)

# %% [markdown]
# In the second example, the opacity corresponds to the inverse normalized entropy of the 6 magnitudes. In other words, opacity is maximal if the most resonant coefficient is the only one with magnitude > 0; whereas the color is white when all coefficients have the same magnitude.

# %%
colors = utils.most_resonant2color(max_coeffs[EXAMPLE_FNAME], inv_entropies[EXAMPLE_FNAME])
by_entropy = True
label = etl.make_wavescape_label(EXAMPLE_FNAME, how, indulge, by_entropy=by_entropy)
ws = wavescapes.Wavescape(colors, width=DEFAULT_FIGURE_SIZE)
ws.draw(aw_per_tick=10, tick_factor=10, label=label, label_size=15) #, subparts_highlighted=[110,134]
path = os.path.join('figures', etl.make_filename(EXAMPLE_FNAME, how, indulge, summary_by_entropy=by_entropy, ext='.png'))
plt.savefig(path)

# %% [markdown]
# # Metrics
#
# In this section, we compile a `pd.DataFrame` containing the results of all the metrics used in the paper. We store both an original version of the metrics where each piece corresponds to a row and the columns the metric results across the coefficients, and a melted version where each tuple (piece, coefficient) is a row and there is one column per metric. Note that the second version only applies to metrics that have one value per coefficient (thus not the inverse coherence and piece fragmentation). The metrics are stored and used for testing in `results/`. Optional plots and tests can be done by adjusting the parameters of the wrapper function `get_metric` that can be found in `etl.py`. 
#
#
# `get_metric`: `mname, pd.DataFrame -> pd.DataFrame` takes the name of the metric to be computed and the dataframe to add the metrics to and returns the updated dataframe. Optionally, it takes the needed `np.array`s to compute each metric, the column names, and specifics for the creation of additional plots and tests. For more details, refer to the documentation of the function.

# %%
# let's start from the available metadata for each piece
metadata_metrics = metadata.copy()

# %% [markdown]
# ## Coefficients’ prevalence
#
# The **prevalence of each coefficient** can be computed using the `mname` 'percentage_resonance' and providing the `max_coeffs` matrix. 
#
# In fact, the prevalence of coefficient $n$ in a piece: $W(n)=1/N \sum_{i \in S(n)} i$ where $N$ is the total number of entries in the matrix, $S(n)$ is the set of the indices of the entries in the summary wavescapes that are attributed to coefficient $n$ (i.e., where coefficient $n$ is the most prominent among the six).

# %% jupyter={"outputs_hidden": false}
# defining column names
cols = [f"percentage_resonances_{i}" for i in range(1,7)]

metadata_metrics = etl.get_metric('percentage_resonance', metadata_metrics, 
                              max_coeffs=max_coeffs, cols=cols, 
                              store_matrix=True, show_plot=True, unified=True,
                              save_name='percentage_resonance', title='Percentage Resonance')

# %%
metadata_metrics[cols].head()

# %% [markdown]
# ### Entropy-weighted coefficients’ prevalence (default)
#
# In order to account for the certainty that a certain coefficient is actually the most resonance, we weigh the previous metric by entropy as follows: $W(n)=1/N \sum_{i \in S(n)} w_i$ where $N$ is the total number of entries in the wavescape, $S(n)$ is the set of the indices of the entries in the summary wavescapes that are attributed to coefficient $n$ (i.e., where coefficient $n$ is the most prominent among the six), and $w_i$ is the weight (opacity) of the $i$-th node in the summary wavescape, in this case, the entropy of $i$.
#
# The **entropy weighted prevalence of each coefficient** can be computed using the `mname` 'percentage_resonance_entropy' and providing the `max_coeffs` matrix and the  `inv_entropies` matrix. 
#

# %% jupyter={"outputs_hidden": false}
cols = [f"percentage_resonances_entropy_{i}" for i in range(1,7)]

metadata_metrics = etl.get_metric('percentage_resonance_entropy', metadata_metrics, 
                              cols=cols,
                              max_coeffs=max_coeffs, inv_entropies=inv_entropies,
                              store_matrix=True, show_plot=True, unified=True,
                              save_name='percentage_resonance_entropy', title='Percentage Resonance (entropy)')

# %%
metadata_metrics[cols].head()

# %% [markdown]
# ## Post-hoc analysis of hierarchical prevalence
# ### Moment of Inertia
#
# Moment of inertia of coefficient $n$ in the summary wavescape: $I(n)=1/N \sum_{i \in S(n)} w_i y_i^2$, where N is the total number of entries in the wavescape, $S(n)$ is the set of the indices of the entries in the summary wavescapes that are attributed to coefficient $n$ (i.e., where coefficient n is the most prominent among the six), $w_i$ is the magnitude of the $i$-th entry in the summary wavescape, and $y_i$ is the vertical coordinate of the $i$-th entry in the summary wavescape.
#
# The **moment of inertia** can be computed using the `mname` 'moment_of_inertia' and providing the `max_coeffs` matrix and the `max_mags` matrix. 
#
#

# %%
cols = [f"moments_of_inertia_{i}" for i in range(1,7)]

metadata_metrics = etl.get_metric('moment_of_inertia', metadata_metrics, 
                              cols=cols,
                              max_coeffs=max_coeffs, max_mags=max_mags,
                              store_matrix=True, show_plot=True, unified=True,
                              save_name='moments_of_inertia', title='Moments of Inertia')

# %%
metadata_metrics[cols].head()

# %% [markdown]
# ## Measure Theoretic Entropy
#
# Measure-theoretic entropy: Let $A={A_1,...,A_k}$ be a (finite) partition of a probability space $(X,P(X),)$: the entropy of the partition $A$ is defined as $H(A)= - \sum_{i} \mu(A_i) \log \mu(A_i)$. We can take $X$ as the support of the wavescape, $A$ as the set of the connected regions in the unified wavescape, and $\mu(Y)=(area-of-Y)/(area-of-X)$ for any subset $Y$ of the wavescape.
#
# The **measure theoretic entropy** can be computed using the `mname` 'partition_entropy' and providing the `max_coeffs` matrix. Note that the measure theoretic entropy is defined as a global metric over the piece. 
#

# %% jupyter={"outputs_hidden": false}
cols = 'partition_entropy'

metadata_metrics = etl.get_metric('partition_entropy', metadata_metrics, 
                              cols=cols,
                              max_coeffs=max_coeffs,
                              store_matrix=True, scatter=True, show_plot=True, unified=True, 
                              save_name='partition_entropy', title='Partition Entropy')

# %%
metadata_metrics[cols].head()

# %% [markdown]
# ## Global prototypicality
#
# The global prototypicality is the slope $P$ of the regression line $V_l ~ P(l/L)+c$, where $c$ is an intercept term. $V_l = \frac{\sum_o M_{o,l}}{n - l}$ is the average maximal magnitude for each hierarchical level as, where $n - l$ is the number of nodes at each hierarchical level and $l$ as the hierarchical level of the summary wavescape.
#
# The **global prototypicality** can be computed using the `mname` 'inverse_coherence' and providing the `max_mags` matrix. Note that the measure global prototypicality also is defined as a global metric over the piece. 
#

# %%
cols = 'inverse_coherence'
metadata_metrics = etl.get_metric('inverse_coherence', metadata_metrics, 
                              cols=cols,
                              max_mags=max_mags,
                              store_matrix=True, show_plot=True, unified=True, scatter=True,
                              save_name='inverse_coherence', title='Inverse Coherence')

# %%
metadata_metrics[cols].head()

# %% [markdown]
# # Storing the final metrics files

# %%
if not os.path.isdir('results'):
    os.makedirs('results')

metadata_metrics.reset_index().to_csv(os.path.join('results','results.csv'), index=False)

# %%
metadata_metrics = metadata_metrics.reset_index()

# %%
# melting the results to be used for testing on coefficient specific metrics

resonances_cols = [f"percentage_resonances_{i}" for i in range(1,7)]
entropy_cols = [f"percentage_resonances_entropy_{i}" for i in range(1,7)]
moi_cols = [f"moments_of_inertia_{i}" for i in range(1,7)]

metadata_res = pd.melt(metadata_metrics, id_vars=['fname', 'length_qb', 'year', 'last_mc', 'partition_entropy', 'inverse_coherence'], value_vars=resonances_cols, 
                       var_name='variable_resonance', value_name='value_resonance')
print(metadata_res.shape)
metadata_res_ent = pd.melt(metadata_metrics, id_vars=['fname', 'length_qb', 'year', 'last_mc', 'partition_entropy', 'inverse_coherence'], value_vars=entropy_cols, 
                       var_name='variable_resonance_entropy', value_name='value_resonance_entropy')
print(metadata_res_ent.shape)
metadata_moi = pd.melt(metadata_metrics, id_vars=['fname', 'length_qb', 'year', 'last_mc', 'partition_entropy', 'inverse_coherence'], value_vars=moi_cols, 
                       var_name='variable_inertia', value_name='value_inertia')
print(metadata_moi.shape)

metadata_melted = pd.concat([metadata_res[['fname', 'length_qb', 'year', 'last_mc', 'partition_entropy', 'inverse_coherence', 'variable_resonance', 'value_resonance']],
                             metadata_res_ent[['variable_resonance_entropy', 'value_resonance_entropy']], 
                             metadata_moi[['variable_inertia', 'value_inertia']]], axis=1)
print(metadata_melted.shape)
metadata_melted.reset_index().to_csv(os.path.join('results','results_melted.csv'), index=False)
metadata_melted.head()
