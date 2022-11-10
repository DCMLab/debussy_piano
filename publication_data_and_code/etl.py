from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import Collection, Dict, Iterator, Optional, TypeVar, Union, overload, Tuple
import os
import re
import gzip
import json
from fractions import Fraction as frac
import multiprocessing

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
# from fractions import Fraction as frac
import pandas as pd
# import numpy as np
# from IPython.display import display, HTML
from wavescapes import apply_dft_to_pitch_class_matrix, build_utm_from_one_row, normalize_dft, circular_hue, Wavescape
from wavescapes.draw import compute_plot_height

from utils import most_resonant, utm2long, long2utm, max_pearsonr_by_rotation, center_of_mass, \
    partititions_entropy, make_plots, \
    make_adj_list, pitch_class_matrix_to_tritone, testing_ols, add_to_metrics, \
    most_resonant_penta_dia, resolve_dir, do_it, make_filename, pitch_class_matrix_to_tritone, max_pearsonr_by_rotation, most_resonant, most_resonant2color, \
    long2utm, make_wavescape_label, NORM_METHODS




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

Normalization: TypeVar = Tuple[str, bool]
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


# def display_wavescapes(wavescape_folder, fname, norm_method, summaries=False, grey=False, rows=2):
#     coeff2path = get_wavescape_fnames(
#         wavescape_folder, norm_method, fname, summaries=summaries, grey=grey)
#     if len(coeff2path) == 0:
#         print("No wavescapes found.")
#         return
#     if summaries:
#         coeff2path = {i: coeff2path[key] for i, key in enumerate(
#             ('mag', 'ent'), 1) if key in coeff2path}
#     total = 2 if summaries else 6
#     assert rows <= total, f"Cannot display {rows} rows for {total} requested elements."
#     cols_per_row = total // rows
#     rows_first_ix = range(1, total+1, cols_per_row)
#     html = "<table>"
#     for row in rows_first_ix:
#         html += "<tr>"
#         for col in range(cols_per_row):
#             i = row + col
#             html += "<td>"
#             if i in coeff2path:
#                 html += f"<img src='{os.path.join(wavescape_folder, coeff2path[i])}' alt='coeff_{i}'/>"
#             html += "</td>"
#         html += "</tr>"
#     html += "</table>"
#     display(HTML(html))


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


# def find_wavescapes(data_folder, norm_params, fname=None, summary_by_ent=None, grey=None, ext='png'):
#     """ Generator function that scans data_folder for particular filenames
#      and yields the paths.

#     Parameters
#     ----------
#     data_folder : str
#         Scan the file names in this directory.
#     norm_params : list of tuple
#         One or several (how, indulge_prototype) pairs.
#     coeff : str, optional
#         If the filenames include a 'c{N}-' component for coefficient N, select N.
#     ext : str, optional
#         The extension of the files to detect.

#     Yields
#     ------
#     (str, int), str, str
#         For each found file matching the critera, return norm_params, debussy_fname, pickled_filepath
#     """
#     norm_params = check_norm_params(norm_params)
#     data_folder = resolve_dir(data_folder)
#     assert os.path.isdir(data_folder), data_folder + \
#         " is not an existing directory."
#     regex = f"^(?P<fname>{fname if fname is not None else '.*?'})"
#     regex += r"-(?:c(?P<coeff>\d)-)?(?P<how>0c|post_norm|max|max_weighted)(?P<indulge_prototype>\+indulge)?"
#     if summary_by_ent is None:
#         regex += r"(?P<summary>-summary-by-(?P<by>ent|mag))?"
#     elif summary_by_ent:
#         regex += r"(?P<summary>-summary-by-ent)"
#     else:
#         regex += r"(?P<summary>-summary-by-mag)"
#     if grey is None:
#         regex += r"(?P<grey>-grey)?"
#     elif grey:
#         regex += r"(?P<grey>-grey)"
#     regex += r"\.(?P<extension>" + ext.lstrip('.').replace('.', r'\.') + ')$'
#     for f in sorted(os.listdir(data_folder)):
#         m = re.search(regex, f)
#         if m is None:
#             continue
#         capture_groups = m.groupdict()
#         does_indulge = capture_groups['indulge_prototype'] is not None
#         params = (capture_groups['how'], does_indulge)
#         if params in norm_params:
#             path = os.path.join(data_folder, f)
#             capture_groups.update(dict(
#                 file=f,
#                 path=path,
#                 does_indulge=does_indulge,
#             ))
#             yield capture_groups


# def get_wavescape_fnames(wavescape_folder, norm_params, fname, summaries=False, grey=False):
#     norm_params = check_norm_params(norm_params)
#     assert len(
#         norm_params) == 1, "This function is meant to fetch images for one type of normalization only."
#     how, indulge = norm_params[0]
#     if indulge:
#         # then we need to get the 6th coefficients of indulge=False
#         norm_params.append((how, False))
#     found = {}
#     for groups in find_wavescapes(wavescape_folder, norm_params, fname, None, grey):
#         if summaries:
#             if groups['by'] is None:
#                 continue
#             key = groups['by']
#         else:
#             if groups['coeff'] is None:
#                 continue
#             if indulge and not (groups['does_indulge'] or groups['coeff'] == '6'):
#                 # this lets the 6th coeff through if indulge is wanted (unindulged are identical)
#                 continue
#             key = int(groups['coeff'])
#         found[key] = groups['file']
#     return found


# def get_correlations(data_folder, long=True):
#     """Returns a dictionary of pickled correlation matrices."""
#     data_folder = resolve_dir(data_folder)
#     result = {}
#     for f in sorted(os.listdir(data_folder)):
#         if f.endswith('-correlations.npy.gz'):
#             fname = f[:-20]
#             corr = load_pickled_file(os.path.join(data_folder, f), long=long)
#             if corr is not None:
#                 result[fname] = corr
#     if len(result) == 0:
#         print(
#             f"No pickled numpy matrices with correct file names found in {data_folder}.")
#     return result


# def get_human_analyses(debussy_repo='.'):
#     dtypes = {
#         'mc': 'Int64',
#         'mn': 'Int64'
#     }
#     conv = {
#         'quarterbeats': frac,
#     }
#     analyses_dir = os.path.join(debussy_repo, 'analyses')
#     analyses_path = os.path.join(analyses_dir, 'analyses.tsv')
#     mc_qb_path = os.path.join(analyses_dir, 'mc_qb.tsv')
#     analyses = pd.read_csv(analyses_path, sep='\t')
#     mc_qb = pd.read_csv(mc_qb_path, sep='\t', dtype=dtypes, converters=conv)
#     mc_qb = parse_interval_index(mc_qb.set_index('qb_interval')).set_index('fnames', append=True).swaplevel()

#     @lru_cache
#     def lesure2measures(L):
#         nonlocal mc_qb
#         L = str(L)
#         candidates = [piece for piece in mc_qb.index.levels[0] if L in piece]
#         if len(candidates) != 1:
#             print(f"Pieces corresponding to L='{L}': {candidates}")
#         return candidates[0], mc_qb.loc[candidates[0]]

#     def mc_pos2qb_pos(df, mc, mc_offset):
#         mc = int(mc)
#         mc_offset = frac(mc_offset) * 4.0
#         try:
#             row = df.set_index('mc').loc[mc]
#         except KeyError:
#             last_mc = df.mc.max()
#             if mc > last_mc + 1:
#                 print(f"L={L} Does not contain MC {mc}.")
#             return pd.NA
#         qb = row['quarterbeats']
#         return qb + mc_offset

#     from_to_pos = []
#     for L, mc_start, mc_start_off, mc_end, mc_end_off in analyses.iloc[:, :5].itertuples(
#             index=False):
#         try:
#             piece, df = lesure2measures(L)
#         except Exception:
#             from_to_pos.append((pd.NA, pd.NA, pd.NA))
#             continue
#         if any(pd.isnull(arg) for arg in (mc_start, mc_start_off, mc_end, mc_end_off)):
#             from_to_pos.append((pd.NA, pd.NA, pd.NA))
#             continue
#         try:
#             start_qb = mc_pos2qb_pos(df, mc_start, mc_start_off)
#         except Exception:
#             print(f"L={L}, mc_start={mc_start}, mc_start_off={mc_start_off}")
#             raise
#         try:
#             end_qb = mc_pos2qb_pos(df, mc_end, mc_end_off)
#         except Exception:
#             print(f"L={L}, mc_start={mc_start}, mc_start_off={mc_start_off}")
#             raise
#         from_to_pos.append((piece, start_qb, end_qb))
#     from_to_df = pd.DataFrame(from_to_pos, index=analyses.index, columns=['fname', 'from_qb', 'to_qb'])
#     analyses = pd.concat([analyses, from_to_df], axis=1)
#     return analyses


# def get_maj_min_coeffs(debussy_repo='.', long=True, get_arg_max=False):
#     """Returns a dictionary of all pitch-class matrices' maximum correlations with a
#     major and a minor profile."""
#     pcms = get_pcms(debussy_repo, long=True)
#     result = {}
#     for fname, pcm in pcms.items():
#         maj_min = np.column_stack([
#             max_pearsonr_by_rotation(
#                 pcm, 'mozart_major', get_arg_max=get_arg_max),
#             max_pearsonr_by_rotation(
#                 pcm, 'mozart_minor', get_arg_max=get_arg_max)
#         ])
#         result[fname] = maj_min if long else long2utm(maj_min)
#     return result


def get_metadata(debussy_repo: str = '..',) -> pd.DataFrame:
    """Reads in the metadata table and enriches it with information from the pieces' median
    recording duration.
    """
    md_path = os.path.join(debussy_repo, 'metadata.tsv')
    metadata = pd.read_csv(md_path, sep='\t', index_col=0)
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

# def get_most_resonant_penta_dia(mag_phase_mx_dict, ninefold_dict, clf):
#     max_coeff, max_mag, inv_entropy = zip(*(most_resonant_penta_dia(mag_phase_mx_dict[piece][..., 0], ninefold_dict[piece], clf)
#                                             for piece in mag_phase_mx_dict.keys()))
#     return (
#         dict(zip(mag_phase_mx_dict.keys(), max_coeff)),
#         dict(zip(mag_phase_mx_dict.keys(), max_mag)),
#         dict(zip(mag_phase_mx_dict.keys(), inv_entropy))
#     )


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


# def get_standard_filename(fname):
#     fname_filter = r"(l\d{3}(?:-\d{2})?(?:_[a-z]+){1,2})"
#     m = re.search(fname_filter, fname)
#     if m is None:
#         return
#     return m.groups(0)[0]


# def get_ttms(debussy_repo='.', long=True):
#     """Returns a dictionary with the results of the tritone detector run on all pitch-class matrices."""
#     pcms = get_pcms(debussy_repo, long=long)
#     return {fname: pitch_class_matrix_to_tritone(pcm) for fname, pcm in pcms.items()}


# def make_feature_vectors(data_folder, norm_params, long=True):
#     """ Return a dictionary with concatenations of magnitude-phase matrices for the
#      selected normalizations with the corresponding correlation matrices.

#     Parameters
#     ----------
#     data_folder : str
#         Folder containing the pickled matrices.
#     norm_params : list of tuple
#         The return format depends on whether you pass one or several (how, indulge_prototypes) pairs.
#     long : bool, optional
#         By default, all matrices are loaded in long format. Pass False to cast to square
#         matrices where the lower left triangle beneath the diagonal is zero.

#     Returns
#     -------
#     dict of str or dict of dict
#         If norm_params is a (list containing a) single tuple, the result is a {debussy_filename -> feature_matrix}
#         dict. If it contains several tuples, the result is a {debussy_filename -> {norm_params -> feature_matrix}}
#     """
#     norm_params = check_norm_params(norm_params)
#     several = len(norm_params) > 1
#     result = defaultdict(dict) if several else dict()
#     mag_phase_mx_dict = get_pickled_magnitude_phase_matrices(data_folder, norm_params, long=True)
#     correl_dict = get_correlations(data_folder, long=True)
#     m_keys, c_keys = set(mag_phase_mx_dict.keys()), set(correl_dict.keys())
#     m_not_c, c_not_m = m_keys.difference(c_keys), c_keys.difference(m_keys)
#     if len(m_not_c) > 0:
#         print(
#             f"No pickled correlations found for the following magnitude-phase matrices: {m_not_c}.")
#     if len(c_not_m) > 0:
#         print(
#             f"No pickled magnitude-phase matrices found for the following correlations: {c_not_m}.")
#     key_intersection = m_keys.intersection(c_keys)
#     for fname in key_intersection:
#         corr = correl_dict[fname]
#         mag_phase = mag_phase_mx_dict[fname]
#         if several:
#             for norm in norm_params:
#                 if not norm in mag_phase:
#                     print(f"No pickled magnitude-phase matrix found for the {norm} normalization "
#                           f"of {fname}.")
#                     continue
#                 mag_phase_mx = mag_phase[norm][..., 0]
#                 features = np.column_stack([mag_phase_mx, corr])
#                 result[fname][norm] = features if long else long2utm(features)
#         else:
#             features = np.column_stack([mag_phase[..., 0], corr])
#             result[fname] = features if long else long2utm(features)
#     return result


def parse_interval_index(df, name='interval'):
    iv_regex = r"\[([0-9]*\.[0-9]+), ([0-9]*\.[0-9]+)\)"
    df = df.copy()
    values = df.index.str.extract(iv_regex).astype(float)
    iix = pd.IntervalIndex.from_arrays(
        values[0], values[1], closed='left', name=name)
    df.index = iix
    return df


def test_dict_keys(dict_keys, metadata):
    found_fnames = metadata.index.isin(dict_keys)
    if found_fnames.all():
        print("Found matrices for all files listed in metadata.tsv.")
    else:
        print(
            f"Couldn't find matrices for the following files:\n{metadata.index[~found_fnames].to_list()}.")


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


def get_metric(metric_type : str, metadata_matrix : pd.DataFrame, cols : list,
               mag_phase_mx_dict : np.array=None,
               max_mags : np.array=None,
               max_coeffs : np.array=None,
               inv_entropies : np.array=None,
               store_matrix : bool=False,
               testing : bool=False,
               show_plot : bool=False,
               save_name : bool=False,
               scatter : bool=False,
               unified : bool=False,
               boxplot : bool=False,
               ordinal : bool=False,
               figsize: tuple=(20, 25),
               title : str=None,
               ordinal_col : str=None,
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
        metric = {fname: np.polyfit((max_mag.shape[1] - np.arange(max_mag.shape[1]))/max_mag.shape[1], np.mean(max_mag, axis=0), 1)[0] for fname, max_mag in
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

def store_pickled_magnitude_phase_matrices(data_folder: str,
                                           norm_params: Collection[Tuple[str, bool]],
                                           dfts: Optional[Dict[str, NDArray]] = None,
                                           debussy_repo: Optional[str] = None,
                                           overwrite: bool = False,
                                           cores: int = 0,
                                           sort: bool = False) -> None:
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
            del(dfts[k])
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


def compute_mag_phase_mx(file_path, dft, how, indulge_prototypes):
    normalized = normalize_dft(dft, how=how, indulge_prototypes=indulge_prototypes)
    with gzip.GzipFile(file_path, "w") as zip_file:
        np.save(file=zip_file, arr=normalized)


def core_msg(cores):
    return "in a for-loop." if cores < 1 else f"using {cores} CPU cores in parallel."


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


def store_wavescapes(wavescape_folder,
                     data_folder,
                     norm_params,
                     coeffs=None,
                     overwrite_standard=False,
                     overwrite_grey=False,
                     overwrite_summary=False,
                     cores=0,
                     sort=False):
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
        coeffs = list(range(1,7))
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
        del(mag_phase_dict[k])
    for key, type2params in key2type2params.items():
        delete_keys = []
        for norm in mag_phase_dict[key]:
            if not any(norm == (how, indulge) for params in type2params.values() for
                       _, how, indulge, *_ in params):
                delete_keys.append(norm)
        for norm in delete_keys:
            del(mag_phase_dict[key][norm])
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


def compute_correlations(file_path, pcm):
    tritones = pitch_class_matrix_to_tritone(pcm)
    maj = max_pearsonr_by_rotation(pcm, 'mozart_major')
    min = max_pearsonr_by_rotation(pcm, 'mozart_minor')
    stacked = np.column_stack([maj, min, tritones])
    with gzip.GzipFile(file_path, "w") as zip_file:
        np.save(file=zip_file, arr=stacked)




def make_wavescape(path, mag_phase_mx, fname, how, indulge, coeff=None, grey=False, by_entropy=False):
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
        colors = colors[...,0]
    ws = Wavescape(colors, width=2286)
    ws.draw(label=label, aw_per_tick=10, tick_factor=10, label_size=20, indicator_size=1.0, tight_layout=False)
    plt.savefig(path)
    plt.close()


def make_all_wavescapes(color_matrix, individual_width, primitive=Wavescape.RHOMBUS_STR, aw_per_tick=None, tick_offset=0, tick_start=0,
                        tick_factor=1., indicator_size=None, add_line=False, subparts_highlighted=None, labels=None, label_size=None):
    """

    Parameters
    ----------

    pc_mat : numpy.array
        A (n, m) array with n pitch class vectors of size m (number of pitch classes, generally 12).
        Each vector corresponds to the pitch classes' summed durations for a segment of the piece.
        They will be summed to form the higher levels in a (n, n, m)-dimensional triangular matrix.

    individual_width: int
        the width in pixel of each individual wavescapes. If no save label is provided,
        then the resulting plot holds all 6 plots and consequently has 3*individual_width
        as width, and a height of two individual wavescapes (the hieght of a wavescape is
        dependent on the width and drawing primitive used)

    save_label: str, optional
        The prefix of the filepath to save each individual plot. If it has the (default)
        value of `None`, then the function produces all six plots into a single 3 by 2 figure
        and don't save it in PNG format (but this can be easily achieved by calling the "saveFig"
        function of matplotlib.pyplot after this one)
        The path can be absolute or relative, however, it should not hold any file extensions at the end,
        as it is generated by this function.
        For example, if the value "bach" is given for this parameter, then the following files will be created:
        bach1.png, bach2.png, bach3.png, bach4.png, bach5.png and bach6.png
        Each number preceding the PNG extension indicates which coefficient is vizualized in the file.
        Default value is None.

    magn_stra: str, optional
        see the doc 'complex_utm_to_ws_utm' for information on this parameter
        Default value is '0c'

    output_rgba:
        see the doc 'complex_utm_to_ws_utm' for information on this parameter
        Default value is '0c'

    primitive: str, optional
        see the doc of the constructor of the class 'Wavescape' for information on this parameter.
        Default value is Wavescape.RHOMBUS_STR (i.e. 'rhombus')

    aw_per_tick: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None (meaning no ticks are drawn)

    tick_offset: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None (meaning ticks numbers start at 0)

    tick_start: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 0

    tick_factor: float, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 1.0

    ignore_magnitude: bool, optional
        Set to True to plot without magnitude.

    ignore_phase: bool, optional
        Set to True to plot without phase.

    indicator_size: boolean, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is True

    add_line: numeric value, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is False

    subparts_highlighted: array of tuples, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None

    label_size: float, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is None (in which case the default
        size of the labels is the width of one individual plot divided by 30)

    Returns
    -------

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
        color_utm = color_matrix[:,:,i]
        w = Wavescape(color_utm, width=individual_width, primitive=primitive)
        ax = fig.add_subplot(2, 3, i+1,
                             aspect='equal')  # TODO: what if fig was not initialised above?
        w.draw(ax=ax, indicator_size=indicator_size, add_line=add_line,
               aw_per_tick=aw_per_tick, tick_offset=tick_offset, tick_start=tick_start,
               tick_factor=tick_factor,
               label=label,
               subparts_highlighted=subparts_highlighted,
               label_size=label_size)
    plt.tight_layout()
    return fig