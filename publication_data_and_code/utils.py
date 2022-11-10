import math
import multiprocessing as mp
import os
import warnings
from functools import lru_cache
from itertools import accumulate, islice
from typing import Callable, Iterable, Collection, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from networkx.algorithms.components import connected_components
from scipy import ndimage
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from wavescapes import legend_decomposition
from wavescapes.color import circular_hue

NORM_METHODS = ['0c', 'post_norm', 'max_weighted', 'max']

########################################
# SQUARE <-> LONG matrix transformations
########################################


def utm2long(utm: NDArray) -> NDArray:
    """(N, N, ...) upper triangle matrix to (N(N+1)/2, ...) long matrix."""
    n, m = utm.shape[:2]
    assert n == m, f"Upper triangular matrix is expected to be square, not ({n}, {m})."
    return utm[np.triu_indices(n)]


def longn2squaren(n: int) -> int:
    """From the length of a long matrix, compute the size of the corresponding square matrix."""
    square_n = np.sqrt(0.25 + 2 * n) - 0.5
    assert square_n % 1. == 0, f"Length {n} does not correspond to an upper triangular matrix in long format."
    return int(square_n)


def long2utm(long: NDArray) -> NDArray:
    """(N(N+1)/2, ...) long matrix to upper triangle matrix where the lower left triangle beneath the diagonal is 0-padded."""
    n, *m = long.shape
    square_n = longn2squaren(n)
    A = np.zeros_like(long, shape=(square_n, square_n, *m))
    A[np.triu_indices(square_n)] = long
    return A


@lru_cache
def longix2squareix(ix: int, n: int, from_to: bool = False) -> Tuple[int, int]:
    """
    Turn the index of a N(N+1)/2 long format UTM (upper triangle matrix) into
    coordinates of a NxN square format UTM.

    Args:
        ix: Index to convert.
        n: Side length of the square matrix.
        from_to:
            By default, the returned coordinates signify (segment_length, last_segment).
            Pass True to return (first_segment, last_segment) instead.

    Returns:
        See `from_to`.
    """
    for x, diag_ix in enumerate(accumulate(range(n, 1, -1))):
        # the accumulated backwards range corresponds to the long indices put on a diagonal
        if ix == diag_ix:
            return x + 1, x + 1
        if ix > diag_ix:
            continue
        y = n - (diag_ix - ix)
        break
    if from_to:
        x = y - x
    return x, y


@lru_cache
def squareix2longix(x, y, n) -> int:
    assert x < n and y < n, "Coordinates need to be smaller than n."
    assert y >= x, f"Coordinates ({x}, {y}) are not within an upper triangular matrix."
    return sum(islice(range(n, -1, -1), x)) + y - x


########################################
# Inspecting complex matrices
########################################

def comp2str(c: complex, dec: int = 2) -> int:
    """Interpret a complex number as magnitude and phase and convert into a human-readable string."""
    magn = np.round(abs(c), dec)
    ang = -round(np.angle(c, True)) % 360
    return f"{magn}+{ang}°"


comp2str_vec = np.vectorize(comp2str)
"""Vectorized version of comp2str() that can be applied to numpy arrays."""


def comp2mag_phase(c: complex, dec: int = 2) -> Tuple[int, int]:
    """Convert a complex DFT coefficient into magnitude and phase."""
    magn = np.round(abs(c), dec)
    ang = np.round(np.angle(c), dec)
    return magn, ang


def get_coeff(dft: NDArray,
              x: int,
              y: int,
              coeff: Optional[int] = None,
              deg: bool = False,
              from_to: bool = False) -> NDArray:
    """View magnitude and phase of a particular point in the matrix.

    Args:
        dft: (NxNx7) complex square matrix or (Nx7) complex long matrix.
        x:
            By default, x designates the row of the wavescape ('length-to notation'). If `from_to` is
            set to True, x is the leftmost index of the selected interval ('from-to notation').
        y: y-1 is the rightmost index of the selected interval.
        coeff: If you want to look at a single coefficient, pass a number between 0 and 6, otherwise all 7 will be returned.
        deg:
            By default, the complex number will be converted into a string containing the rounded
            magnitude and the angle in degrees. Pass false to get the raw complex number.
        from_to:
            See `x`.

    Returns:
        Shape 1 or 7 depending on `coeff`, dtype depends on `deg`.
    """
    assert dft.ndim in (2, 3), f"2D or 3D, not {dft.ndim}D"
    if dft.ndim == 2:
        is_long = True
        long_n, n_coeff = dft.shape
        n = longn2squaren(long_n)
        xs, ys = n, n
    else:
        is_long = False
        xs, ys, n_coeff = dft.shape
    if coeff is not None:
        assert 0 <= coeff < n_coeff, f"0 <= coeff < {n_coeff}"
    assert 0 <= x < xs, f"0 <= x < {xs}; received x = {x}"
    assert 0 <= y < ys, f"0 <= y < {ys}; received y = {y}"
    if from_to:
        x = y - x
    if is_long:
        ix = squareix2longix(x, y, n)
        result = dft[ix]
    else:
        result = dft[x, y]
    if coeff is not None:
        result = result[[coeff]]
    if deg:
        return comp2str_vec(result)[:, None]
    return np.apply_along_axis(comp2mag_phase, -1, result).T


########################################
# Summary wavescapes
########################################

def most_resonant(mag_mx: NDArray, add_one: bool = False) -> Tuple[NDArray, NDArray, NDArray]:
    """ Takes a NxNx6 square or N(N+1)/2x6 long matrix of magnitudes and computes three matrices of shape NxN or N(N+1)/2
    for plotting summary wavescapes.

    Args:
        mag_mx: Matrix of magnitudes in square or long format.
        add_one: By default the most resonant coefficients are numbers between 0 and 5. Pass True to return numbers between 1 and 6 instead.

    Returns:
        most_resonant_coeff (NxN) matrix where indices between 0 and 5 or 1 and 6 (depending on ``add_one``) correspond to the six DFT coefficients 1 through 6.
        maximum_magnitude (NxN) matrix containing the most resonant coefficients' magnitudes.
        inverse_entropy (NxN) matrix where each value corresponds to the inverse normalized entropy of the 6 magnitudes.
    """
    is_square = mag_mx.ndim == 3
    utm_max = np.max(mag_mx, axis=-1)
    utm_argmax = np.argmax(mag_mx, axis=-1)
    if add_one:
        utm_argmax = np.triu(utm_argmax + 1)
    if is_square:
        # so we don't apply entropy to zero-vectors
        mag_mx = utm2long(mag_mx)
    # entropy and np.log have same base e
    with warnings.catch_warnings():
        # suppress warnings for 0-vectors representing rests in the music
        warnings.simplefilter("ignore")
        utm_entropy = 1 - (entropy(mag_mx, axis=-1) / np.log(mag_mx.shape[-1]))
    utm_entropy = np.nan_to_num(utm_entropy) # replace nan by 0
    utm_entropy = MinMaxScaler().fit_transform(
        utm_entropy.reshape(-1, 1)).reshape(-1)
    if is_square:
        utm_entropy = long2utm(utm_entropy)
    return utm_argmax, utm_max, utm_entropy


def most_resonant2color(max_coeff: NDArray,
                        opacity: NDArray,
                        hue_segments: int = 6,
                        **kwargs) -> NDArray:
    """ Computes a color matrix by dividing the hue circle into ``hue_segments`` segments, selecting these colors according to
    the indices in ``max_coeff``, and weighting the opacity by the ``opacity`` matrix.

    Args:
        max_coeff: Array of integers within [0, hue_segments).
        opacity: Corresponding opacity values.
        hue_segments: Into how many equidistant segments to divide the hue circle.
        **kwargs: Keyword arguments passed on to wavescapes.circular_hue()

    Returns:

    """
    if hue_segments is None:
        hue_segments = max_coeff.max()
    hue_segment = math.tau / hue_segments
    phase = max_coeff * hue_segment
    mag_dims, phase_dims = opacity.ndim, phase.ndim
    assert mag_dims == phase_dims, f"Both arrays should have the same dimensionality"
    if mag_dims > 1:
        mag_phase_mx = np.dstack([opacity, phase])
    else:
        mag_phase_mx = np.column_stack([opacity, phase])
    return circular_hue(mag_phase_mx, **kwargs)


def store_color_legend(file_path: str = None) -> None:
    """ Produce a circular legend for the most_resonant summary wavescapes.

    Args:
        file_path: Where to store the legend.
    """
    def make_pcv(position_of_one):
        return [0] * position_of_one + [1] + [0] * (11 - position_of_one)
    legend = {f'c{i + 1}': (make_pcv(6 - i), [2]) for i in range(0, 6)}
    legend_decomposition(legend, width=5, single_img_coeff=2)
    if file_path is not None:
        plt.savefig(file_path)


########################################
# Measures
########################################

PITCH_CLASS_PROFILES = {'mozart_major': [0.20033700035703508,
                                         0.010812613711830977,
                                         0.11399209672667372,
                                         0.012104110819714938,
                                         0.13638736763981654,
                                         0.1226311293358654,
                                         0.018993520966402003,
                                         0.20490464831336042,
                                         0.014611863068643751,
                                         0.07414111247856302,
                                         0.011351150687477073,
                                         0.07973338589461738],
                        'mozart_minor': [0.1889699424221723,
                                         0.008978237532936468,
                                         0.11060533806574259,
                                         0.1308781295283801,
                                         0.011630786793101157,
                                         0.11019324778803881,
                                         0.029590747199631288,
                                         0.21962043820162988,
                                         0.07742468998919953,
                                         0.012129908077215472,
                                         0.020386372564054407,
                                         0.07959216183789804]}


@lru_cache
def get_precomputed_rotations(key):
    assert key in PITCH_CLASS_PROFILES, f"Key needs to be one of {list(PITCH_CLASS_PROFILES.keys())}," \
                                        f"not {key}."
    b = np.array(PITCH_CLASS_PROFILES[key])
    n = b.shape[0]
    B_rotated_cols = np.array([np.roll(b, i) for i in range(n)]).T
    B = B_rotated_cols - b.mean()
    b_std = b.std()
    return B, b_std, n


def max_pearsonr_by_rotation(A, b, get_arg_max=False):
    """ For every row in A return the maximum person correlation from all transpositions of b

    Parameters
    ----------
    A : np.array
      (n,m) matrix where the highest correlation will be found for each row.
    b : np.array or str
      (m,) vector to be rolled m times to find the highest possible correlation.
      You can pass a key to use precomputed values for the profiles contained in
      PITCH_CLASS_PROFILES
    get_arg_max : bool, optional
      By default, an (n,) vector with the maximum correlation per row of A is returned.
      Set to True to retrieve an (n,2) matrix where the second column has the argmax,
      i.e. the number of the rotation producing the highest correlation.

    Returns
    -------
    np.array
      (n,) or (n,2) array
    """
    if isinstance(b, str):
        B, b_std, n = get_precomputed_rotations(b)
    else:
        b = b.flatten()
        n = b.shape[0]
        B_rotated_cols = np.array([np.roll(b, i) for i in range(n)]).T
        B = B_rotated_cols - b.mean()
        b_std = b.std()
    assert n == A.shape[1], f"Profiles in A have length {A.shape[1]} but the profile to roll and" \
                            f"compare has length {n}."
    norm_by = A.std(axis=1, keepdims=True) * b_std * n
    all_correlations = (A - A.mean(axis=1, keepdims=True)) @ B
    all_correlations = np.divide(all_correlations, norm_by, out=np.zeros_like(
        all_correlations), where=norm_by > 0)
    if get_arg_max:
        return np.stack([all_correlations.max(axis=1), all_correlations.argmax(axis=1)]).T
    return all_correlations.max(axis=1)


def pitch_class_matrix_to_tritone(pc_mat):
    """
    This functions takes a list of N pitch class distributions,
    modelised by a matrix of float numbers, and apply the 
    DFT individually to all the pitch class distributions.

    Args:
        pc_mat (np.array): pitch class matrix

    Returns:
        np.array: matrix of tritone presence
    """
    coeff_nmb = 6
    res = np.linalg.norm(np.multiply(pc_mat/np.linalg.norm(pc_mat, axis=1).reshape(-1, 1),
                                     np.roll(pc_mat/np.linalg.norm(pc_mat,
                                             axis=1).reshape(-1, 1), coeff_nmb, axis=-1)
                                     )[..., :coeff_nmb], axis=-1)
    return res


########################################
# metrics
########################################

def center_of_mass(utm):
    """Computes the vertical center of mass for each coefficient in the wavescape

    Args:
        utm (np.array): array containing the 6 wavescapes

    Returns:
        list: 6 vertical centers of mass (normalized by the height of the wavescape)
    """
    vcoms = []
    shape_y, shape_z = np.shape(utm)[1:3]
    for i in range(shape_z):
        utm_interest = utm[:, :, i]
        vcoms.append(ndimage.measurements.center_of_mass(
            utm_interest)[0] / shape_y)
    return vcoms


def add_to_adj_list(adj_list, a, b):
    """Util function to compute the partition entropy

    Args:
        adj_list (list): list containing the pair of adjacent nodes
        a (int): node 1
        b (int): node 2
    """
    adj_list.setdefault(a, []).append(b)
    adj_list.setdefault(b, []).append(a)


def make_adj_list(max_coeff):
    """Creates an adjacency list from the matrix of most resonant coefficients
       in order to convert the matrix to a network.

    Args:
        max_coeff (np.array): matrix of most resonant coefficients

    Returns:
        list: adjacency list
    """
    adj_list = {}

    utm_index = np.arange(0, max_coeff.shape[0] * max_coeff.shape[1]).reshape(max_coeff.shape[0],
                                                                              max_coeff.shape[1])
    for i in range(len(max_coeff)):
        for j in range(len(max_coeff)):
            if (j < len(max_coeff[i]) - 1) and (max_coeff[i][j] == max_coeff[i][j + 1]):
                add_to_adj_list(adj_list, utm_index[i][j], utm_index[i][j + 1])
            if i < len(max_coeff[i]) - 1:
                for x in range(max(0, j - 1), min(len(max_coeff[i + 1]), j + 2)):
                    if (max_coeff[i][j] == max_coeff[i + 1][x]):
                        add_to_adj_list(
                            adj_list, utm_index[i][j], utm_index[i + 1][x])
    return adj_list


def partititions_entropy(adj_list):
    """Computes the entropy of the different connected components
       of the network obtained from the adjacency list.

    Args:
        adj_list (list): adjacency list

    Returns:
        int: normalized entropy of the partitions
    """
    G = nx.Graph(adj_list)
    components = connected_components(G)
    lengths = [len(x) / G.size() for x in components]
    ent = entropy(lengths) / entropy([1] * G.size())
    return ent


########################################
# utils for metrics
########################################


def make_plots(metadata_metrics : pd.DataFrame, save_name : str, title : str, cols : list,
               figsize : tuple=(20, 25), 
               unified : bool=False, scatter : bool=False, 
               boxplot : bool=False, ordinal: bool=False, 
               ordinal_col: str='years_ordinal'):
    """Creates custom plots to show the evolution of the metrics.

    Args:
        metadata_metrics (pd.DataFrame): df containing the already computed metrics
        save_name (str): name used for saving the visualization
        title (str): title of the visualization
        cols (list): list of columns plotted
        figsize (tuple, optional): size of the plot. Defaults to (20,25).
        unified (bool, optional): whether the metrics for each coefficient should be plotted in only one axis. Defaults to False.
        scatter (bool, optional): whether to scatter the points in the unified plot. Defaults to False.
        boxplot (bool, optional): to use boxplots instead of regplots (suggested for ordinal plots). Defaults to False.
        ordinal (bool, optional): whether to show the time evolution as an ordinal number. Defaults to False.
        ordinal_col (str, optional): the column that should be used as ordinal values. Defaults to 'years_ordinal'.
    """

    if unified:
        fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    else:
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        axs = axs.flatten()

    fig.suptitle(title)
    if type(cols) == str:
        len_cols = 2
    else:
        len_cols = len(cols) + 1

    for i in range(1, len_cols):
        if type(cols) == str:
            col = cols
        else:
            if len_cols > 7:
                if i == 1:
                    continue
            col = cols[i-1]
        
        if ordinal:
            x_col = ordinal_col
        else:
            x_col = 'year'

        if not unified:
            if len_cols > 7:
                ax = axs[i-2]
                ax.set_title(f"Coefficient {i-2}")
            else:
                ax = axs[i-1]
                ax.set_title(f"Coefficient {i-1}")
            if boxplot:
                sns.boxplot(x=x_col, y=col, data=metadata_metrics, ax=ax)
            else:
                sns.regplot(x=x_col, y=col, data=metadata_metrics, ax=ax)
        else:
            if boxplot:
                sns.boxplot(x=x_col, y=col, data=metadata_metrics)
            else:
                p = sns.regplot(x=x_col, y=col, data=metadata_metrics,
                            scatter=scatter, label=f"Coefficient {i}")
        
            plt.legend()
    if unified and not boxplot:
        p.set_ylabel(col[:-2])
    
    if not os.path.isdir('figures/'):
        os.makedirs('figures/')
    plt.savefig(f'figures/{save_name}.png')


def testing_ols(metadata_matrix : pd.DataFrame, cols : list, 
                ordinal : bool=False, melted : bool=True,
                ordinal_col : str='years_ordinal'):
    """Function used to test the predictiveness of each measure with respect to 
       the time period.

    Args:
        metadata_matrix (pd.DataFrame): df containing all the metrics
        cols (list): list of columns to be tested
        ordinal (bool, optional): whether to show the time evolution as an ordinal number. Defaults to False.
        ordinal_col (str, optional): the column that should be used as ordinal values. Defaults to 'years_ordinal'.
    """

    scaler = StandardScaler()
    if type(cols) != str:
        metadata_sm = metadata_matrix[metadata_matrix[cols[0]].notnull()]
        if not melted:  
            metadata_sm[cols] = scaler.fit_transform(metadata_sm[cols])
    else:
        metadata_sm = metadata_matrix[metadata_matrix[cols].notnull()]
        metadata_sm[cols] = scaler.fit_transform(
            np.array(metadata_sm[cols]).reshape(-1, 1))
        cols = [cols]

    
    if melted:
        metadata_sm = metadata_sm.reset_index()
        metadata_sm['fname'] = metadata_sm['index']
        metadata_sm = pd.melt(metadata_sm, id_vars=['fname', 'length_qb', 'year', 'last_mc'], value_vars=cols)
        results = smf.ols(formula='value ~ year * C(variable) + last_mc + 1 ', data=metadata_sm).fit()
        print('testing results')
        print(results.summary())
        

    else:
        for col in cols:
            if col[-1] not in ['1', '2', '3']:
                results = smf.ols(formula=f'{col} ~ year + last_mc + 1 ', data=metadata_sm).fit()
                print('testing results')
                print(results.summary())

    if ordinal:
        print(ordinal_col) # use ordinal col
    

def add_to_metrics(metrics_df : pd.DataFrame, dict_metric : dict, name_metrics):
    """Function to add the newly computed metrics to the dataframe.

    Args:
        metrics_df (pd.DataFrame): df containing already computed metrics
        dict_metric (dict): name:metric dictionary
        name_metrics (list/str): list of columns names (str if only one column) 

    Returns:
        _type_: _description_
    """
    if type(name_metrics) == str:
        if name_metrics in metrics_df.columns:
            metrics_df = metrics_df.drop(columns=name_metrics)
        df_tmp = pd.Series(dict_metric, name=name_metrics)
    else:
        if name_metrics[0] in metrics_df.columns:
            metrics_df = metrics_df.drop(columns=name_metrics)
        df_tmp = pd.DataFrame(dict_metric).T
        df_tmp.columns = name_metrics
    metrics_df = metrics_df.merge(df_tmp, left_index=True, right_index=True)
    return metrics_df


def do_it(func: Callable, params: Iterable[tuple], n: Optional[int] = None, cores: int = 0) -> Collection:
    """Call a function on a list of argument tuples, potentially parallelized, and return the result.

    Args:
        func: Function to map on ``params``.
        params:
            Collection or Iterator of tuples where each tuple corresponds to a set of arguments passed to ``func``.
            The arguments need to be in the right order to match the signature of ``func``.
        n: Number of function calls for displaying the progress bar. Needed only if ``params`` is an Iterator.
        cores: On how many CPU cores to perform the operation in parallel. Defaults to 0, meaning no parallelization.

    Returns:
        _description_
    """
    if n is None:
        n = len(list(params))
    if cores == 0:
        return [func(*p) for p in tqdm(params, total=n)]
    pool = mp.Pool(cores)
    result = pool.starmap(func, tqdm(params, total=n))
    pool.close()
    pool.join()
    return result


def make_filename(fname, how, indulge_prototypes, coeff=None, summary_by_entropy=None, ext=None) -> str:
    result = fname
    if coeff is not None:
        result += f"-c{coeff}"
    result += f"-{how}"
    if indulge_prototypes:
        result += "+indulge"
    if summary_by_entropy is not None:
        result += "-summary-by-ent" if summary_by_entropy else "-summary-by-mag"
    if ext is not None:
        result += ext
    return result


def resolve_dir(d):
    """ Resolves '~' to HOME directory and turns ``d`` into an absolute path.
    """
    if d is None:
        return None
    if '~' in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)




def make_wavescape_label(fname: str, how: str, indulge: bool, coeff: Optional[int] = None, by_entropy: Optional[bool] = None) -> str:
    label = f"{fname}: "
    normalization = f"Normalization: {how}{'+' if indulge else ''}"
    if coeff is None:
        if by_entropy is not None:
            # summary wavescape
            label += f"summary\n{normalization}\n"
            if by_entropy:
                label += "Opacity: inverse entropy"
            else:
                label += "Opacity: magnitude"
        else:
            label += f"all coefficients\n{normalization}"
    else:
        label += f"c{coeff}\n{normalization}"
    return label
