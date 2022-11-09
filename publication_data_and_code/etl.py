from collections import defaultdict
from functools import lru_cache
from itertools import product
import os
import re
import gzip
import json
from fractions import Fraction as frac
from xmlrpc.client import boolean
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from wavescapes import apply_dft_to_pitch_class_matrix, build_utm_from_one_row

from utils import most_resonant, utm2long, long2utm, max_pearsonr_by_rotation, center_of_mass, \
    partititions_entropy, make_plots, \
    make_adj_list, pitch_class_matrix_to_tritone, testing_ols, add_to_metrics, \
    most_resonant_penta_dia


NORM_METHODS = ['0c', 'post_norm', 'max_weighted', 'max']


def get_dfts(debussy_repo='.', long=True):
    pcvs = get_pcvs(debussy_repo)
    return {fname: apply_dft_to_pitch_class_matrix(pcv, long=long) for fname, pcv in pcvs.items()}


def load_pickled_file(path, long=True):
    """Unzips and loads the file and returns it in long or square format."""
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


def get_mag_phase_mx(data_folder, norm_params, long=True):
    """ Search data_folder for pickled magnitude_phase matrices corresponding to one
    or several normalization methods and load them into a dictionary.

    Parameters
    ----------
    data_folder : str
        Directory to scan for files.
    norm_params : list of tuple
        The return format depends on whether you pass one or several (how, indulge_prototypes) pairs.
    long : bool, optional
        By default, all matrices are loaded in long format. Pass False to cast to square
        matrices where the lower left triangle beneath the diagonal is zero.

    Returns
    -------
    dict of str or dict of dict
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


def check_norm_params(norm_params):
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


def display_wavescapes(wavescape_folder, fname, norm_method, summaries=False, grey=False, rows=2):
    coeff2path = get_wavescape_fnames(
        wavescape_folder, norm_method, fname, summaries=summaries, grey=grey)
    if len(coeff2path) == 0:
        print("No wavescapes found.")
        return
    if summaries:
        coeff2path = {i: coeff2path[key] for i, key in enumerate(
            ('mag', 'ent'), 1) if key in coeff2path}
    total = 2 if summaries else 6
    assert rows <= total, f"Cannot display {rows} rows for {total} requested elements."
    cols_per_row = total // rows
    rows_first_ix = range(1, total+1, cols_per_row)
    html = "<table>"
    for row in rows_first_ix:
        html += "<tr>"
        for col in range(cols_per_row):
            i = row + col
            html += "<td>"
            if i in coeff2path:
                html += f"<img src='{os.path.join(wavescape_folder, coeff2path[i])}' alt='coeff_{i}'/>"
            html += "</td>"
        html += "</tr>"
    html += "</table>"
    display(HTML(html))


def find_pickles(data_folder, norm_params, ext='npy.gz'):
    """ Generator function that scans data_folder for particular filenames
     and yields the paths.

    Parameters
    ----------
    data_folder : str
        Scan the file names in this directory.
    norm_params : list of tuple
        One or several (how, indulge_prototype) pairs.
    ext : str, optional
        The extension of the files to detect.

    Yields
    ------
    (str, int), str, str
        For each found file matching the critera, return norm_params, debussy_fname, pickled_filepath
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


def find_wavescapes(data_folder, norm_params, fname=None, summary_by_ent=None, grey=None, ext='png'):
    """ Generator function that scans data_folder for particular filenames
     and yields the paths.

    Parameters
    ----------
    data_folder : str
        Scan the file names in this directory.
    norm_params : list of tuple
        One or several (how, indulge_prototype) pairs.
    coeff : str, optional
        If the filenames include a 'c{N}-' component for coefficient N, select N.
    ext : str, optional
        The extension of the files to detect.

    Yields
    ------
    (str, int), str, str
        For each found file matching the critera, return norm_params, debussy_fname, pickled_filepath
    """
    norm_params = check_norm_params(norm_params)
    data_folder = resolve_dir(data_folder)
    assert os.path.isdir(data_folder), data_folder + \
        " is not an existing directory."
    regex = f"^(?P<fname>{fname if fname is not None else '.*?'})"
    regex += r"-(?:c(?P<coeff>\d)-)?(?P<how>0c|post_norm|max|max_weighted)(?P<indulge_prototype>\+indulge)?"
    if summary_by_ent is None:
        regex += r"(?P<summary>-summary-by-(?P<by>ent|mag))?"
    elif summary_by_ent:
        regex += r"(?P<summary>-summary-by-ent)"
    else:
        regex += r"(?P<summary>-summary-by-mag)"
    if grey is None:
        regex += r"(?P<grey>-grey)?"
    elif grey:
        regex += r"(?P<grey>-grey)"
    regex += r"\.(?P<extension>" + ext.lstrip('.').replace('.', r'\.') + ')$'
    for f in sorted(os.listdir(data_folder)):
        m = re.search(regex, f)
        if m is None:
            continue
        capture_groups = m.groupdict()
        does_indulge = capture_groups['indulge_prototype'] is not None
        params = (capture_groups['how'], does_indulge)
        if params in norm_params:
            path = os.path.join(data_folder, f)
            capture_groups.update(dict(
                file=f,
                path=path,
                does_indulge=does_indulge,
            ))
            yield capture_groups


def get_wavescape_fnames(wavescape_folder, norm_params, fname, summaries=False, grey=False):
    norm_params = check_norm_params(norm_params)
    assert len(
        norm_params) == 1, "This function is meant to fetch images for one type of normalization only."
    how, indulge = norm_params[0]
    if indulge:
        # then we need to get the 6th coefficients of indulge=False
        norm_params.append((how, False))
    found = {}
    for groups in find_wavescapes(wavescape_folder, norm_params, fname, None, grey):
        if summaries:
            if groups['by'] is None:
                continue
            key = groups['by']
        else:
            if groups['coeff'] is None:
                continue
            if indulge and not (groups['does_indulge'] or groups['coeff'] == '6'):
                # this lets the 6th coeff through if indulge is wanted (unindulged are identical)
                continue
            key = int(groups['coeff'])
        found[key] = groups['file']
    return found


def get_correlations(data_folder, long=True):
    """Returns a dictionary of pickled correlation matrices."""
    data_folder = resolve_dir(data_folder)
    result = {}
    for f in sorted(os.listdir(data_folder)):
        if f.endswith('-correlations.npy.gz'):
            fname = f[:-20]
            corr = load_pickled_file(os.path.join(data_folder, f), long=long)
            if corr is not None:
                result[fname] = corr
    if len(result) == 0:
        print(
            f"No pickled numpy matrices with correct file names found in {data_folder}.")
    return result


def get_human_analyses(debussy_repo='.'):
    dtypes = {
        'mc': 'Int64',
        'mn': 'Int64'
    }
    conv = {
        'quarterbeats': frac,
    }
    analyses_dir = os.path.join(debussy_repo, 'analyses')
    analyses_path = os.path.join(analyses_dir, 'analyses.tsv')
    mc_qb_path = os.path.join(analyses_dir, 'mc_qb.tsv')
    analyses = pd.read_csv(analyses_path, sep='\t')
    mc_qb = pd.read_csv(mc_qb_path, sep='\t', dtype=dtypes, converters=conv)
    mc_qb = parse_interval_index(mc_qb.set_index('qb_interval')).set_index('fnames', append=True).swaplevel()

    @lru_cache
    def lesure2measures(L):
        nonlocal mc_qb
        L = str(L)
        candidates = [piece for piece in mc_qb.index.levels[0] if L in piece]
        if len(candidates) != 1:
            print(f"Pieces corresponding to L='{L}': {candidates}")
        return candidates[0], mc_qb.loc[candidates[0]]

    def mc_pos2qb_pos(df, mc, mc_offset):
        mc = int(mc)
        mc_offset = frac(mc_offset) * 4.0
        try:
            row = df.set_index('mc').loc[mc]
        except KeyError:
            last_mc = df.mc.max()
            if mc > last_mc + 1:
                print(f"L={L} Does not contain MC {mc}.")
            return pd.NA
        qb = row['quarterbeats']
        return qb + mc_offset

    from_to_pos = []
    for L, mc_start, mc_start_off, mc_end, mc_end_off in analyses.iloc[:, :5].itertuples(
            index=False):
        try:
            piece, df = lesure2measures(L)
        except Exception:
            from_to_pos.append((pd.NA, pd.NA, pd.NA))
            continue
        if any(pd.isnull(arg) for arg in (mc_start, mc_start_off, mc_end, mc_end_off)):
            from_to_pos.append((pd.NA, pd.NA, pd.NA))
            continue
        try:
            start_qb = mc_pos2qb_pos(df, mc_start, mc_start_off)
        except Exception:
            print(f"L={L}, mc_start={mc_start}, mc_start_off={mc_start_off}")
            raise
        try:
            end_qb = mc_pos2qb_pos(df, mc_end, mc_end_off)
        except Exception:
            print(f"L={L}, mc_start={mc_start}, mc_start_off={mc_start_off}")
            raise
        from_to_pos.append((piece, start_qb, end_qb))
    from_to_df = pd.DataFrame(from_to_pos, index=analyses.index, columns=['fname', 'from_qb', 'to_qb'])
    analyses = pd.concat([analyses, from_to_df], axis=1)
    return analyses


def get_maj_min_coeffs(debussy_repo='.', long=True, get_arg_max=False):
    """Returns a dictionary of all pitch-class matrices' maximum correlations with a
    major and a minor profile."""
    pcms = get_pcms(debussy_repo, long=True)
    result = {}
    for fname, pcm in pcms.items():
        maj_min = np.column_stack([
            max_pearsonr_by_rotation(
                pcm, 'mozart_major', get_arg_max=get_arg_max),
            max_pearsonr_by_rotation(
                pcm, 'mozart_minor', get_arg_max=get_arg_max)
        ])
        result[fname] = maj_min if long else long2utm(maj_min)
    return result


def get_metadata(debussy_repo='.'):
    md_path = os.path.join(debussy_repo, 'metadata.tsv')
    dur_path = os.path.join(
        debussy_repo, 'durations/spotify_median_durations.json')
    metadata = pd.read_csv(md_path, sep='\t', index_col=1)
    print(f"Metadata for {metadata.shape[0]} files.")
    with open(os.path.join(debussy_repo, 'durations/spotify_median_durations.json'), 'r', encoding='utf-8') as f:
        durations = json.load(f)
    idx2key = pd.Series(metadata.index.str.split('_').map(
        lambda l: l[0][1:] if l[0] != 'l000' else l[1]), index=metadata.index)
    fname2duration = idx2key.map(durations).rename('median_recording')
    fname2year = (
        (metadata.composed_end + metadata.composed_start) / 2).rename('year')
    qb_per_minute = (60 * metadata.length_qb_unfolded /
                     fname2duration).rename('qb_per_minute')
    sounding_notes_per_minute = (
        60 * metadata.all_notes_qb / fname2duration).rename('sounding_notes_per_minute')
    sounding_notes_per_qb = (
        metadata.all_notes_qb / metadata.length_qb_unfolded).rename('sounding_notes_per_qb')
    return pd.concat([
        metadata,
        fname2year,
        fname2duration,
        qb_per_minute,
        sounding_notes_per_qb,
        sounding_notes_per_minute
    ], axis=1)


def get_most_resonant(mag_phase_mx_dict):
    max_coeff, max_mag, inv_entropy = zip(*(most_resonant(mag_phase_mx[..., 0])
                                            for mag_phase_mx in mag_phase_mx_dict.values()))
    return (
        dict(zip(mag_phase_mx_dict.keys(), max_coeff)),
        dict(zip(mag_phase_mx_dict.keys(), max_mag)),
        dict(zip(mag_phase_mx_dict.keys(), inv_entropy))
    )

def get_most_resonant_penta_dia(mag_phase_mx_dict, ninefold_dict, clf):
    max_coeff, max_mag, inv_entropy = zip(*(most_resonant_penta_dia(mag_phase_mx_dict[piece][..., 0], ninefold_dict[piece], clf)
                                            for piece in mag_phase_mx_dict.keys()))
    return (
        dict(zip(mag_phase_mx_dict.keys(), max_coeff)),
        dict(zip(mag_phase_mx_dict.keys(), max_mag)),
        dict(zip(mag_phase_mx_dict.keys(), inv_entropy))
    )


@lru_cache
def get_pcms(debussy_repo='.', long=True):
    pcvs = get_pcvs(debussy_repo, pandas=False)
    return {fname: build_utm_from_one_row(pcv, long=long) for fname, pcv in pcvs.items()}


@lru_cache
def get_pcvs(debussy_repo, pandas=False):
    pcvs_path = os.path.join(debussy_repo, 'pcvs',
                             'debussy-1.0q_sliced-w0.5-pc-pcvs.tsv')
    pcvs = pd.read_csv(pcvs_path, sep='\t', index_col=[0, 1, 2])
    pcv_dfs = {fname: pcv_df.reset_index(
        level=[0, 1], drop=True) for fname, pcv_df in pcvs.groupby(level=1)}
    if pandas:
        pcv_dfs = {k: parse_interval_index(v) for k, v in pcv_dfs.items()}
    if not pandas:
        pcv_dfs = {k: v.to_numpy() for k, v in pcv_dfs.items()}
    return pcv_dfs


def get_standard_filename(fname):
    fname_filter = r"(l\d{3}(?:-\d{2})?(?:_[a-z]+){1,2})"
    m = re.search(fname_filter, fname)
    if m is None:
        return
    return m.groups(0)[0]


def get_ttms(debussy_repo='.', long=True):
    """Returns a dictionary with the results of the tritone detector run on all pitch-class matrices."""
    pcms = get_pcms(debussy_repo, long=long)
    return {fname: pitch_class_matrix_to_tritone(pcm) for fname, pcm in pcms.items()}


def make_feature_vectors(data_folder, norm_params, long=True):
    """ Return a dictionary with concatenations of magnitude-phase matrices for the
     selected normalizations with the corresponding correlation matrices.

    Parameters
    ----------
    data_folder : str
        Folder containing the pickled matrices.
    norm_params : list of tuple
        The return format depends on whether you pass one or several (how, indulge_prototypes) pairs.
    long : bool, optional
        By default, all matrices are loaded in long format. Pass False to cast to square
        matrices where the lower left triangle beneath the diagonal is zero.

    Returns
    -------
    dict of str or dict of dict
        If norm_params is a (list containing a) single tuple, the result is a {debussy_filename -> feature_matrix}
        dict. If it contains several tuples, the result is a {debussy_filename -> {norm_params -> feature_matrix}}
    """
    norm_params = check_norm_params(norm_params)
    several = len(norm_params) > 1
    result = defaultdict(dict) if several else dict()
    mag_phase_mx_dict = get_mag_phase_mx(data_folder, norm_params, long=True)
    correl_dict = get_correlations(data_folder, long=True)
    m_keys, c_keys = set(mag_phase_mx_dict.keys()), set(correl_dict.keys())
    m_not_c, c_not_m = m_keys.difference(c_keys), c_keys.difference(m_keys)
    if len(m_not_c) > 0:
        print(
            f"No pickled correlations found for the following magnitude-phase matrices: {m_not_c}.")
    if len(c_not_m) > 0:
        print(
            f"No pickled magnitude-phase matrices found for the following correlations: {c_not_m}.")
    key_intersection = m_keys.intersection(c_keys)
    for fname in key_intersection:
        corr = correl_dict[fname]
        mag_phase = mag_phase_mx_dict[fname]
        if several:
            for norm in norm_params:
                if not norm in mag_phase:
                    print(f"No pickled magnitude-phase matrix found for the {norm} normalization "
                          f"of {fname}.")
                    continue
                mag_phase_mx = mag_phase[norm][..., 0]
                features = np.column_stack([mag_phase_mx, corr])
                result[fname][norm] = features if long else long2utm(features)
        else:
            features = np.column_stack([mag_phase[..., 0], corr])
            result[fname] = features if long else long2utm(features)
    return result


def parse_interval_index(df, name='iv'):
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


def resolve_dir(d):
    """ Resolves '~' to HOME directory and turns ``d`` into an absolute path.
    """
    if d is None:
        return None
    if '~' in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)



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
        _type_: _description_
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
