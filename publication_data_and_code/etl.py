import gzip
import json
import multiprocessing
import os
import re
from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import Collection, Dict, Iterator, Optional, TypeVar, Union, overload, Tuple, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from wavescapes import apply_dft_to_pitch_class_matrix, build_utm_from_one_row, normalize_dft, circular_hue, Wavescape
from wavescapes.draw import compute_plot_height

from utils import utm2long, center_of_mass, partititions_entropy, make_plots, make_adj_list, testing_ols, add_to_metrics, \
    resolve_dir, do_it, make_filename, pitch_class_matrix_to_tritone, max_pearsonr_by_rotation, most_resonant, most_resonant2color, \
    long2utm, make_wavescape_label, NORM_METHODS

Normalization: TypeVar = Tuple[str, bool]

def check_norm_params(norm_params: Union[Normalization, Collection[Normalization]]) -> Collection[Normalization]:
    """If the argument is a tuple, turn it into a list of one tuple. Then check if
    the tuples correspond to valid normalization parameters."""
    int2norm = {i: (how, indulge) for i, (indulge, how) in enumerate(
        product((False, True), ('0c', 'post_norm', 'max_weighted', 'max')))}
    if isinstance(norm_params, tuple) or isinstance(norm_params, int):
        norm_params = [norm_params]
    norm_params = [int2norm[p] if isinstance(
        p, int) else p for p in norm_params]
    for t in norm_params:
        assert len(
            t) == 2, f"norm_params need to be (how, indulge_prototypes) pairs, not {t}"
        assert t[0] in NORM_METHODS, f"how needs to be one of {NORM_METHODS}, not {t[0]}"
    return norm_params


def compute_correlations(file_path: str, pcm: NDArray) -> None:
    """Compute and store the Pearson correlation coefficients of all pitch class vectors with a major and minor profile reflecting all
    piano sonatas by W.A. Mozart."""
    tritones = pitch_class_matrix_to_tritone(pcm)
    maj = max_pearsonr_by_rotation(pcm, 'mozart_major')
    min = max_pearsonr_by_rotation(pcm, 'mozart_minor')
    stacked = np.column_stack([maj, min, tritones])
    with gzip.GzipFile(file_path, "w") as zip_file:
        np.save(file=zip_file, arr=stacked)


def compute_mag_phase_mx(file_path: str,
                         dft: NDArray,
                         how: str,
                         indulge_prototypes: bool) -> None:
    """Compute a magnitude-phase matrix from a DFT matrix applying a normalization methode, and pickle the result to disk."""
    normalized = normalize_dft(dft, how=how, indulge_prototypes=indulge_prototypes)
    with gzip.GzipFile(file_path, "w") as zip_file:
        np.save(file=zip_file, arr=normalized)


def core_msg(cores):
    return "in a for-loop." if cores < 1 else f"using {cores} CPU cores in parallel."


def find_pickles(data_folder: str,
                 norm_params: Collection[Normalization],
                 ext: str = 'npy.gz') -> Iterator[Tuple[Tuple[str, bool], str, str]]:
    """
    Generator function that scans data_folder for particular filenames and yields the paths.

    Args:
        data_folder: Scan the file names in this directory.
        norm_params: One or several (how, indulge_prototype) pairs.
        ext: The extension of the files to detect. Defaults to 'npy.gz'.

    Yields:
        For each found file matching the critera, yield (norm_params, debussy_fname, pickled_filepath.
    """
    norm_params = check_norm_params(norm_params)
    data_folder = resolve_dir(data_folder)
    assert os.path.isdir(data_folder), data_folder + \
                                       " is not an existing directory."
    ext_reg = ext.lstrip('.').replace('.', r'\.') + ')$'
    data_regex = r"^(?P<fname>.*?)-(?P<how>0c|post_norm|max|max_weighted)(?P<indulge_prototype>\+indulge)?\.(?P<extension>" + ext_reg
    for f in sorted(os.listdir(data_folder)):
        m = re.search(data_regex, f)
        if m is None:
            continue
        capture_groups = m.groupdict()
        does_indulge = capture_groups['indulge_prototype'] is not None
        params = (capture_groups['how'], does_indulge)
        if params in norm_params:
            path = os.path.join(data_folder, f)
            yield params, capture_groups['fname'], path


def get_dfts(debussy_repo: str = '..', long: bool = False) -> Dict[str, NDArray]:
    """ For each piece, compute the Discrete Fourier Transform of the aggregated pitch class vectors corresponding to all segmentations.

    Args:
        debussy_repo: Path to the local clone of DCMLab/debussy_piano.
        long:
            By default, the dictionary values will be 3-dimensional upper triangle matrices with shape (N, N, 7) where N is the number of slices in a piece.
            If set to True, they will be reduced to 2-dimensional (N*(N+1)/2, 7) matrices, eliminating the 0-padding.

    Returns:
        `{fname -> np.array}` dict of `(NxNx7)` complex matrices.
    """
    pcvs = get_pcvs(debussy_repo)
    return {fname: apply_dft_to_pitch_class_matrix(pcv, long=long) for fname, pcv in pcvs.items()}


def get_magnitude_phase_matrices(dfts: Dict[str, NDArray],
                                 data_folder: str,
                                 norm_params: Union[Normalization, Collection[Normalization]],
                                 parallelized: bool = True,
                                 long: bool = False,
                                 ):
    """
    Apply one or several normalizations to the given DFT matrices, turning the complex numbers for each coefficient
    to magnitude-phase pairs. The shape changes from (N, N, 7) to (N, N, 6, 2).
    The results are pickled to ``data_folder`` and when pickled files are found,
    the values are not being re-computed.

    Args:
        dfts: The {fname -> dft} dictionary where dft are numpy arrays with dimensions (N, N, 7)
        data_folder: Where to store the pickled matrices.
        norm_params: One or several (how, indulge) tuples.
        parallelized:
            By default, the computation is performed on all available CPU cores in parallel.
            Pass False to prevent that.
        long:
            By default, the matrices are 4-dimensional upper triangle matrices with shape (N, N, 6, 2) where N is the number of slices in a piece.
            If set to True, they will be reduced to 2-dimensional (N*(N+1)/2, 6, 2) matrices, eliminating the 0-padding.


    Returns:
        _description_
    """
    norm_params = check_norm_params(norm_params)
    cores = multiprocessing.cpu_count() if parallelized else 0
    store_pickled_magnitude_phase_matrices(data_folder=data_folder,
                                           norm_params=norm_params,
                                           dfts=dfts,
                                           overwrite=False,
                                           cores=cores,
                                           sort=True)
    result = get_pickled_magnitude_phase_matrices(data_folder=data_folder,
                                                  norm_params=norm_params,
                                                  long=long)
    return result


def get_metadata(debussy_repo: str = '..', ) -> pd.DataFrame:
    """Reads in the metadata table and enriches it with information from the pieces' median
    recording duration.
    """
    md_path = os.path.join(debussy_repo, 'concatenated_metadata.tsv')
    metadata = pd.read_csv(md_path, sep='\t', index_col="fname").sort_index()
    fname2year = ((metadata.composed_end + metadata.composed_start) / 2).rename('year')
    metadata = pd.concat([metadata, fname2year], axis=1)
    print(f"Metadata for {metadata.shape[0]} files.")
    dur_path = os.path.join(debussy_repo, 'publication_data_and_code', 'durations', 'spotify_median_durations.json')
    if not os.path.isfile(dur_path):
        print(f"{dur_path} not found.")
        return metadata
    with open(dur_path, 'r', encoding='utf-8') as f:
        durations = json.load(f)
    idx2key = pd.Series(metadata.index.str.split('_').map(
        lambda l: l[0][1:] if l[0] != 'l000' else l[1]), index=metadata.index)
    fname2duration = idx2key.map(durations).rename('median_recording')
    qb_per_minute = (60 * metadata.length_qb_unfolded /
                     fname2duration).rename('qb_per_minute')
    sounding_notes_per_minute = (
            60 * metadata.all_notes_qb / fname2duration).rename('sounding_notes_per_minute')
    sounding_notes_per_qb = (
            metadata.all_notes_qb / metadata.length_qb_unfolded).rename('sounding_notes_per_qb')
    return pd.concat([
        metadata,
        fname2duration,
        qb_per_minute,
        sounding_notes_per_qb,
        sounding_notes_per_minute
    ], axis=1)


def get_metric(metric_type: str, metadata_matrix: pd.DataFrame, cols: list,
               mag_phase_mx_dict: np.array = None,
               max_mags: np.array = None,
               max_coeffs: np.array = None,
               inv_entropies: np.array = None,
               store_matrix: bool = False,
               testing: bool = False,
               show_plot: bool = False,
               save_name: bool = False,
               scatter: bool = False,
               unified: bool = False,
               boxplot: bool = False,
               ordinal: bool = False,
               figsize: tuple = (20, 25),
               title: str = None,
               ordinal_col: str = None,
               ):
    """Wrapper that allows to compute the desired metric on the whole data, store it in a
       dataframe, produce the desired visualization and print the desired test.

    Args:
        metric_type (str): _description_
        metadata_matrix (pd.DataFrame): _description_
        cols (list): _description_
        mag_phase_mx_dict (np.array, optional): _description_. Defaults to None.
        max_mags (np.array, optional): _description_. Defaults to None.
        max_coeffs (np.array, optional): _description_. Defaults to None.
        inv_entropies (np.array, optional): _description_. Defaults to None.
        store_matrix (bool, optional): _description_. Defaults to False.
        testing (bool, optional): _description_. Defaults to False.
        show_plot (bool, optional): _description_. Defaults to False.
        save_name (bool, optional): _description_. Defaults to False.
        scatter (bool, optional): _description_. Defaults to False.
        unified (bool, optional): _description_. Defaults to False.
        boxplot (bool, optional): _description_. Defaults to False.
        ordinal (bool, optional): _description_. Defaults to False.
        figsize (tuple, optional): _description_. Defaults to (20, 25).
        title (str, optional): _description_. Defaults to None.
        ordinal_col (str, optional): _description_. Defaults to None.

    Returns:
        dict/pd.DataFrame: either the dictionary name:metric or the dataframe (if store_matrix=True)
    """
    if metric_type == 'center_of_mass':
        metric = {fname: center_of_mass(
            mag_phase_mx[..., 0]) for fname, mag_phase_mx in mag_phase_mx_dict.items()}
    elif metric_type == 'center_of_mass_most_resonant':
        metric = {fname: np.divide(np.array(
            [
                (
                        max_mags[fname][max_coeff == i] *
                        np.divide(np.indices(max_mags[fname].shape)[0], max_coeff.shape[1])[
                            max_coeff == i]

                ).sum()
                for i in range(len(cols))
            ]),
            max_coeff.shape[0] * max_coeff.shape[1] / 2)
            for fname, max_coeff in max_coeffs.items()

        }
    elif metric_type == 'mean_resonance':
        metric = {fname: np.mean(mag_phase_mx[..., 0], axis=(
            0, 1)) for fname, mag_phase_mx in mag_phase_mx_dict.items()}
    elif metric_type == 'moment_of_inertia':
        metric = {fname: np.divide(np.array(
            [
                (
                        max_mags[fname][max_coeff == i] *
                        np.square(
                            np.divide(np.indices(max_mags[fname].shape)[0], max_coeff.shape[1]))[
                            max_coeff == i]

                ).sum()
                for i in range(len(cols))
            ]),
            max_coeff.shape[0] * max_coeff.shape[1] / 2)
            for fname, max_coeff in max_coeffs.items()

        }
    elif metric_type == 'percentage_resonance':
        metric = {fname: np.divide(np.array([(max_coeff == i).sum() for i in range(len(cols))]),
                                   max_coeff.shape[0] * max_coeff.shape[1] / 2) for fname, max_coeff in
                  max_coeffs.items()}

    elif metric_type == 'percentage_resonance_entropy':
        metric = {fname: np.divide(
            np.array([(inv_entropies[fname] * (max_coeff == i)).sum()
                      for i in range(len(cols))]),
            max_coeff.shape[0] * max_coeff.shape[1] / 2) for fname, max_coeff in max_coeffs.items()}
    elif metric_type == 'partition_entropy':
        metric = {fname: partititions_entropy(make_adj_list(max_coeff)) for fname, max_coeff in
                  max_coeffs.items()}
    elif metric_type == 'inverse_coherence':
        metric = {fname: np.polyfit((max_mag.shape[1] - np.arange(max_mag.shape[1])) / max_mag.shape[1], np.mean(max_mag, axis=0), 1)[0] for fname, max_mag in
                  max_mags.items()}
    else:
        return 'Metric not implemented, choose among: center_of_mass, mean_resonance, moment_of_inertia, percentage_resonance, percentage_resonance_entropy, partition_entropy, inverse_coherence'

    if type(cols) != str:
        print_check_examples(metric, metadata_matrix, 'l000_etude')

    if store_matrix:
        metadata_matrix = add_to_metrics(metadata_matrix, metric, cols)
        if show_plot:
            make_plots(metadata_matrix, save_name, title, cols,
                       figsize, unified, scatter, boxplot, ordinal, ordinal_col)
        if testing:
            testing_ols(metadata_matrix, cols, ordinal=ordinal,
                        ordinal_col=ordinal_col)
        return metadata_matrix
    else:
        return metric


def get_most_resonant(mag_phase_mx_dict: Dict[str, NDArray]) -> Tuple[Dict[str, NDArray],
                                                                      Dict[str, NDArray],
                                                                      Dict[str, NDArray]]:
    """
    Applies most_resonant() to each array in the given magnitude-phase matrix and returns the
    results as three dictionaries.

    Args:
        mag_phase_mx_dict:
            A {fname -> magnitude-phrase matrix} dict where matrices are normalized magnitude-phase
            matrices with shape (N, N, 6, 2).

    Returns:
        {fname -> most_resonant_coeff} dict with (NxN) matrices where indices between 0 and 5 correspond to the six DFT coefficients 1 through 6.
        {fname -> maximum_magnitude} dict with (NxN) matrices containing the most resonant coefficients' magnitudes.
        {fname -> inverse_entropy} dict with (NxN) matrices where each value corresponds to the inverse normalized entropy of the 6 magnitudes.
    """
    max_coeff, max_mag, inv_entropy = zip(*(most_resonant(mag_phase_mx[..., 0])
                                            for mag_phase_mx in mag_phase_mx_dict.values()))
    return (
        dict(zip(mag_phase_mx_dict.keys(), max_coeff)),
        dict(zip(mag_phase_mx_dict.keys(), max_mag)),
        dict(zip(mag_phase_mx_dict.keys(), inv_entropy))
    )


@lru_cache
def get_pcms(debussy_repo: str = '.', long: bool = False) -> Dict[str, NDArray]:
    """From the pitch-class vectors for each piece, compute a matrix representing the aggregated PCVs for all possible segmentations.

    If `long=False` the (NxNx12) square matrices contain values only in the upper right triangle, with the lower left beneath the diagonal padded with zeros.
    The values are arranged such that row 0 correponds to the original PCV, row 1 the aggregated PCVs for all segments of length = 2 quarter notes, etc.
    For getting the segment reaching from slice 3 to 5 (including), i.e. length 3, the coordinates are `(2, 5)`
    (think x = 'length - 1' and y = index of the last slice included).

    Args:
        debussy_repo: Path to the local clone of DCMLab/debussy_piano.
        long:
            By default, the dictionary values will be 3-dimensional upper triangle matrices with shape (N, N, 12) where N is the number of slices in a piece.
            If set to True, they will be reduced to (N*(N+1)/2, 12) matrices, eliminating the 0-padding.

    Returns:
        An `{fname -> np.array}` dictionary where each matrix contains the aggregated PCVs for all segments that make up a piece.

    """
    pcvs = get_pcvs(debussy_repo, pandas=False)
    return {fname: build_utm_from_one_row(pcv, long=long) for fname, pcv in pcvs.items()}


@overload
def get_pcvs(debussy_repo: str = '..', pandas: bool = True) -> Dict[str, pd.DataFrame]:
    ...


@lru_cache
def get_pcvs(debussy_repo: str = '..', pandas: bool = False) -> Dict[str, NDArray]:
    """Load the pre-computed pitch class vectors for all pieces.

    Args:
        debussy_repo: Path to the local clone of DCMLab/debussy_piano.
        pandas: Pass True to retrieve a DataFrame with and IntervalIndex rather than a dictionary. Defaults to False.

    Returns:
        An `{fname -> PCVs}` dictionary where each `(NX12)` array or DataFrame contains the absolute durations (expressed in quarter nots) of the 12
        chromatic pitch classes for the `N` slices of length = 1 quarter note that make up the piece `fname`. If `pandas=True`,
        the IntervalIndex reflects each slice's position in the piece.
    """
    pcvs_path = os.path.join(debussy_repo, 'publication_data_and_code',
                             'debussy-1.0q_sliced-w0.5-pc-pcvs.tsv')
    pcvs = pd.read_csv(pcvs_path, sep='\t', index_col=[0, 1, 2])
    pcv_dfs = {fname: pcv_df.reset_index(
        level=[0, 1], drop=True) for fname, pcv_df in pcvs.groupby(level=1)}
    if pandas:
        pcv_dfs = {k: parse_interval_index(v) for k, v in pcv_dfs.items()}
    if not pandas:
        pcv_dfs = {k: v.to_numpy() for k, v in pcv_dfs.items()}
    return pcv_dfs


@overload
def get_pickled_magnitude_phase_matrices(data_folder, norm_params: Collection[Normalization], long: bool = False) -> Dict[str, Dict[str, NDArray]]:
    ...


def get_pickled_magnitude_phase_matrices(data_folder, norm_params: Normalization, long: bool = False) -> Dict[str, NDArray]:
    """Search data_folder for pickled magnitude_phase matrices corresponding to one
    or several normalization methods and load them into a dictionary. If not found, they are
    re-computed and pickled under the given folder.

    Args:
        data_folder: Directory to scan for pickled files.
        norm_params: The return format depends on whether you pass one or several (how, indulge_prototypes) pairs.
        long:
            By default, the dictionary values will be 4-dimensional upper triangle matrices with shape (N, N, 6, 2) where N is the number of slices in a piece.
            If set to True, they will be reduced to 3-dimensional (N*(N+1)/2, 6, 2) matrices, eliminating the 0-padding.

    Returns:
        If norm_params is a (list containing a) single tuple, the result is a {debussy_filename -> pickle_filepath}
        dict. If it contains several tuples, the result is a {debussy_filename -> {norm_params -> pickle_filepath}}
    """
    norm_params = check_norm_params(norm_params)
    several = len(norm_params) > 1
    result = defaultdict(dict) if several else dict()
    for norm, fname, path in find_pickles(data_folder, norm_params):
        mag_phase_mx = load_pickled_file(path, long=long)
        if mag_phase_mx is None:
            continue
        if several:
            result[fname][norm] = mag_phase_mx
        else:
            result[fname] = mag_phase_mx
    if len(result) == 0:
        print(
            f"No pickled numpy matrices with correct file names found in {data_folder}.")
    return dict(result)


def load_pickled_file(path: str, long: bool = True) -> NDArray:
    """Unzips and loads the file and returns it in long or square format.

    Args:
        path: Path to a pickled matrix.
        long:
            By default, the matrices are upper triangle matrices with shape (N, N, ...) where N is the number of slices in a piece.
            If set to True, they will be reduced to 2-dimensional (N*(N+1)/2, ...) matrices, eliminating the 0-padding.

    Returns:
        The pickled numpy array.
    """
    try:
        with gzip.GzipFile(path, "r") as zip_file:
            matrix = np.load(zip_file, allow_pickle=True)
    except Exception as e:
        print(f"Loading pickled {path} failed with exception\n{e}")
        return None
    n, m = matrix.shape[:2]
    if long and matrix.ndim > 2 and n == m:
        matrix = utm2long(matrix)
    if not long and n != m:
        matrix = long2utm(matrix)
    return matrix


def make_all_wavescapes(color_matrix: NDArray,
                        individual_width: int,
                        labels: Optional[Union[str, Collection[str]]] = None,
                        primitive: Literal['rhombus', 'diamond', 'hexagon'] = 'rhombus',
                        **kwargs) -> None:
    """ From a given NxNx6 matrix of HTML colors, create and store a figure containing one wavescape for each of the 6 coefficients.


    Args:
        color_matrix: An NxNx6 upper triangle matrix containing HTML colors.
        individual_width:
            the width in pixel of each individual wavescapes. If no save label is provided,
            then the resulting plot holds all 6 plots and consequently has 3*individual_width
            as width, and a height of two individual wavescapes (the hieght of a wavescape is
            dependent on the width and drawing primitive used)
        labels: One label to be added to each subplot with the coefficient appended, or else a list of 6 labels.
        primitive: The shape of the unicolor shapes making up a wavescape.
        **kwargs: Keyword arguments passed to wavescapes.Wavescape.draw()
    """
    dpi = 96  # (most common dpi values for computers' screen)
    total_width = (3.1 * individual_width) / dpi
    total_height = (2.1 * compute_plot_height(individual_width, color_matrix.shape[0],
                                              primitive)) / dpi
    fig = plt.figure(figsize=(total_width, total_height), dpi=dpi)
    label = None
    for i in range(6):
        if labels is not None:
            if isinstance(labels, str):
                label = f"{labels}\n-c{i}"
            else:
                label = labels[i]
        color_utm = color_matrix[:, :, i]
        w = Wavescape(color_utm, width=individual_width, primitive=primitive)
        ax = fig.add_subplot(2, 3, i + 1,
                             aspect='equal')  # TODO: what if fig was not initialised above?
        w.draw(ax=ax, label=label, **kwargs)
    plt.tight_layout()


def make_wavescape(path: str,
                   mag_phase_mx: NDArray,
                   fname: str,
                   how: str,
                   indulge: bool,
                   coeff: Optional[int] = None,
                   grey: bool = False,
                   by_entropy: bool = False) -> None:
    """ Store a wavescape for one coefficient or a summary wavescape on disk.

    Args:
        path: File path of the figure to be created including the file extension.
        mag_phase_mx: (NxNx6x2) Magnitude-phase matrix.
        fname: Name of the piece (for the in-figure label).
        how: Normalization method (for the in-figure label).
        indulge: Wether or not the indulge_prototypes normalization has been applied (for the in-figure label).
        coeff:
            Defaults to None, in which case a summary wavescape will be stored. Pass an integer between 1 and 6 to
            plot one of the coefficients instead.
        grey: Set to True if you want a greyscale wavescape.
        by_entropy:
            Has an effect only if coeff is None and decides if the opacity of the colors should reflect the magnitude of
            the most resonant coefficient (default) or the inverse normalized entropy of all six magnitudes (pass True).
    """
    label = make_wavescape_label(fname=fname, how=how, indulge=indulge, coeff=coeff, by_entropy=by_entropy)
    if coeff is None:
        # summary wavescape
        if by_entropy:
            max_coeff, _, opacity = most_resonant(mag_phase_mx[..., 0])
        else:
            max_coeff, opacity, _ = most_resonant(mag_phase_mx[..., 0])
        colors = most_resonant2color(max_coeff, opacity)
    else:
        colors = circular_hue(mag_phase_mx[..., coeff - 1, :], output_rgba=True, ignore_phase=grey)
    colors = long2utm(colors)
    if colors.shape[-1] == 1:
        colors = colors[..., 0]
    ws = Wavescape(colors, width=2286)
    ws.draw(label=label, aw_per_tick=10, tick_factor=10, label_size=20, indicator_size=1.0, tight_layout=False)
    plt.savefig(path)
    plt.close()


def parse_interval_index(df, name='interval'):
    """Returns a copy of the DataFrame where the index has been replaced with an IntervalIndex."""
    iv_regex = r"\[([0-9]*\.[0-9]+), ([0-9]*\.[0-9]+)\)"
    df = df.copy()
    values = df.index.str.extract(iv_regex).astype(float)
    iix = pd.IntervalIndex.from_arrays(
        values[0], values[1], closed='left', name=name)
    df.index = iix
    return df


def print_check_examples(metric_results, metadata, example_filename):
    """Wrapper around test_dict_keys that shows some info about the metrics computed.

    Args:
        metric_results (dict): name:metric dictionary
        metadata (pd.DataFrame): df containing the metadata
        example_filename (str): name of an example file
    """
    test_dict_keys(metric_results, metadata)
    print(f"The example center of mass list has len {len(metric_results[example_filename])}.")
    print('Example results', metric_results[example_filename])


def store_correlations(debussy_repo, data_folder, overwrite=False, cores=0, sort=False):
    print("Computing pitch-class-vector triangles...", end=' ')
    pcms = get_pcms(debussy_repo, long=True)
    print('DONE')
    pcms = {os.path.join(data_folder, fname + '-correlations.npy.gz'): pcm
            for fname, pcm in pcms.items()}
    if not overwrite:
        pcms = {path: pcm for path, pcm in pcms.items() if not os.path.isfile(path)}
    n = len(pcms)
    if n == 0:
        print("No new correlation matrices to be computed.")
        return
    params = list(pcms.items())
    if sort:
        params = sorted(params, key=lambda t: t[1].shape[0])
    print(f"Computing correlation matrices for {n} pieces {core_msg(cores)}...")
    _ = do_it(compute_correlations, params, n=n, cores=cores)


def store_pickled_magnitude_phase_matrices(data_folder: str,
                                           norm_params: Collection[Tuple[str, bool]],
                                           dfts: Optional[Dict[str, NDArray]] = None,
                                           debussy_repo: Optional[str] = None,
                                           overwrite: bool = False,
                                           cores: int = 0,
                                           sort: bool = False) -> None:
    """ Compute normalized magnitude-phase matrices from the given DFT matrices and pickle them to disk.

    Args:
        data_folder: Where to pickle the matrices.
        norm_params: For which normalization methods to compute magnitude-phase matrices.
        dfts: Either pass a dictionary with DFT matrices, or
        debussy_repo: The path to the local clone of DCMLab/debussy_piano.
        overwrite: Pass True if you want to recompute and overwrite existing pickled matrices.
        cores: On how many CPU cores to perform the operation in parallel. Defaults to 0, meaning no parallelization.
        sort: Pass True if you want to compute matrices for shorter pieces first.

    Returns:

    """
    assert (dfts is None) + (debussy_repo is None) < 2, "I need DFTs or the path to the clone of DCMLab/debussy_piano."
    if dfts is None:
        print("Creating DFT matrices...", end=' ')
        dfts = get_dfts(debussy_repo, long=True)
        print('DONE')
    fpath2params = {os.path.join(data_folder, make_filename(k, how, indulge, ext='.npy.gz')): (k, how, indulge)
                    for k, (how, indulge) in product(dfts.keys(), norm_params)}

    if overwrite:
        pieces = len(dfts)
    else:
        print("Checking for existing files to be skipped...", end=' ')
        fpath2params = {fpath: params
                        for fpath, params in fpath2params.items()
                        if not os.path.isfile(fpath)}
        used_keys = set(params[0] for params in fpath2params.values())
        pieces = len(used_keys)
        delete_keys = [k for k in dfts.keys() if k not in used_keys]
        for k in delete_keys:
            del (dfts[k])
        print('DONE')
    n_runs = len(fpath2params)
    if n_runs == 0:
        print("No new magnitude-phase matrices to be computed.")
        return
    params = [(path, dfts[key], how, indulge)
              for path, (key, how, indulge) in fpath2params.items()]
    if sort:
        params = sorted(params, key=lambda t: t[1].shape[0])
    print(f"Computing {n_runs} magnitude-phase matrices for {pieces} pieces {core_msg(cores)}...")
    _ = do_it(compute_mag_phase_mx, params, n=n_runs, cores=cores)


def store_wavescapes(wavescape_folder: str,
                     data_folder: str,
                     norm_params: Union[Normalization, Collection[Normalization]],
                     coeffs: Optional[Collection[int]] = None,
                     overwrite_standard: bool = False,
                     overwrite_grey: bool = False,
                     overwrite_summary: bool = False,
                     cores: int = 0,
                     sort: bool = False):
    """ Store wavescapes for the given parameters on disk.

    Args:
        wavescape_folder: Where to store the figures.
        data_folder: Where to find pickled magnitude-phase matrices for the given normalization methods.
        norm_params: For which normalization methods to create wavescapes.
        coeffs: For which coefficients to create wavescapes (defaults to all 6).
        overwrite_standard: Whether to create the standard wavescapes showing a single coefficient, overwriting existing files.
        overwrite_grey: Whether to create greyscale wavescapes showing a single coefficient, overwriting existing files.
        overwrite_summary: Whether to create summary wavescapes showing a single coefficient, overwriting existing files.
        cores:
            On how many CPU cores to perform the operation in parallel. Defaults to 0, meaning no parallelization. For long
            pieces (N > 1000) parallel computation may fill up your memory and make your computer crash.
        sort:
            Pass True if you want to compute wavescapes for shorter pieces first. E.g. to gradually reduce the number
            of ``cores`` when it comes to longer pieces.
    """
    print("Loading magnitude-phase matrices...", end=' ')
    mag_phase_dict = get_pickled_magnitude_phase_matrices(data_folder, norm_params)
    first_norm = norm_params[0]
    if len(norm_params) == 1:
        # unify dict structure
        mag_phase_dict = {k: {first_norm: v} for k, v in mag_phase_dict.items()}
    print("DONE")
    if sort:
        print("Sorting by length...", end=' ')
        mag_phase_dict = dict(sorted(mag_phase_dict.items(), key=lambda t: t[1][first_norm].shape[0]))
        print("DONE")
    if coeffs is None:
        coeffs = list(range(1, 7))
    print("Assemble file paths and names based on parameters...", end=' ')
    fpath2params = {
        'standard': {os.path.join(wavescape_folder,
                                  make_filename(k,
                                                how,
                                                indulge,
                                                coeff
                                                )
                                  ): (k, how, indulge, coeff)
                     for k, (how, indulge), coeff in product(mag_phase_dict.keys(),
                                                             norm_params,
                                                             coeffs)
                     if not (indulge and coeff == 6)},
        'summary': {os.path.join(wavescape_folder,
                                 make_filename(k,
                                               how,
                                               indulge,
                                               summary_by_entropy=by_entropy,
                                               ext='.png'
                                               )
                                 ): (k, how, indulge, by_entropy)
                    for k, (how, indulge), by_entropy in product(mag_phase_dict.keys(), norm_params, (False, True))},
    }
    fpath2params['grey'] = {path + '-grey.png': params for path, params in fpath2params['standard'].items()}
    fpath2params['standard'] = {path + '.png': params for path, params in fpath2params['standard'].items()}
    # check for existing files that are not to be overwritten
    ws_types = tuple(fpath2params.keys())
    overwrite = dict(zip(ws_types, (overwrite_standard, overwrite_grey, overwrite_summary)))
    for ws_type in ws_types:
        if not overwrite[ws_type]:
            fpath2params[ws_type] = {path: params
                                     for path, params in fpath2params[ws_type].items()
                                     if not os.path.isfile(path)}
    print("DONE")
    print("Getting settled...")
    key2type2params = defaultdict(lambda: {t: [] for t in ws_types})
    for ws_type, path2params in fpath2params.items():
        for path, params in path2params.items():
            key, *p = params
            key2type2params[key][ws_type].append((path, *p))
    # delete unneeded mag_phase_matrices to save RAM
    delete_keys = [k for k in mag_phase_dict.keys() if k not in key2type2params]
    for k in delete_keys:
        del (mag_phase_dict[k])
    for key, type2params in key2type2params.items():
        delete_keys = []
        for norm in mag_phase_dict[key]:
            if not any(norm == (how, indulge) for params in type2params.values() for
                       _, how, indulge, *_ in params):
                delete_keys.append(norm)
        for norm in delete_keys:
            del (mag_phase_dict[key][norm])
    pieces = len(key2type2params)
    parameters = []
    for key, type2params in key2type2params.items():
        for ws_type, params in type2params.items():
            for p in params:
                if ws_type == 'summary':
                    path, how, indulge, by_entropy = p
                    parameters.append((
                        path,
                        mag_phase_dict[key][(how, indulge)],
                        key,
                        how,
                        indulge,
                        None,
                        False,
                        by_entropy
                    ))
                else:
                    path, how, indulge, coeff = p
                    grey = ws_type == 'grey'
                    parameters.append((
                        path,
                        mag_phase_dict[key][(how, indulge)],
                        key,
                        how,
                        indulge,
                        coeff,
                        grey,
                        False
                    ))
    n = len(parameters)
    print(f"Computing {n} wavescapes for {pieces} pieces {core_msg(cores)}...")
    _ = do_it(make_wavescape, parameters, n=n, cores=cores)


def test_dict_keys(dict_keys: Collection[str], metadata: pd.DataFrame) -> None:
    """Check if one of the computations yielded results for all pieces contained in the index of the metadata DataFrame.
    The result of the test is printed JFYI.

    Args:
        dict_keys: Names of the pieces for which something was computed.
        metadata: A Dataframe where the index corresponds to all pieces for which something was supposed to be computed.
    """
    found_fnames = metadata.index.isin(dict_keys)
    if found_fnames.all():
        print("Found matrices for all files listed in metadata.tsv.")
    else:
        print(
            f"Couldn't find matrices for the following files:\n{metadata.index[~found_fnames].to_list()}.")



