# pip install git+https://github.com/DCMLab/wavescapes.git@3a69d34 tqdm
import os, argparse
from itertools import product
import multiprocessing as mp

from etl import store_pickled_magnitude_phase_matrices, store_correlations, store_wavescapes
from utils import resolve_dir, NORM_METHODS


def check_and_create(d):
    """ Turn input into an existing, absolute directory path.
    """
    if not os.path.isdir(d):
        d = resolve_dir(os.path.join(os.getcwd(), d))
        if not os.path.isdir(d):
            if input(d + ' does not exist. Create? (y|n)') == "y":
                os.mkdir(d)
            else:
                raise argparse.ArgumentTypeError(d + ' needs to be an existing directory')
    return resolve_dir(d)


def check_dir(d):
    if not os.path.isdir(d):
        d = resolve_dir(os.path.join(os.getcwd(), d))
        if not os.path.isdir(d):
            raise argparse.ArgumentTypeError(d + ' needs to be an existing directory')
    return resolve_dir(d)


def main(args):
    store_pickled_magnitude_phase_matrices(data_folder=args.data,
                                           norm_params=args.normalization,
                                           debussy_repo=args.repo,
                                           overwrite=args.magphase,
                                           cores=args.cores,
                                           sort=args.sort,
                                           )
    store_correlations(args.repo,
                       data_folder=args.data,
                       overwrite=args.correlations,
                       cores=args.cores,
                       sort=args.sort,
                       )
    if args.wavescapes is not None:
        store_wavescapes(
            args.wavescapes,
            data_folder=args.data,
            norm_params=args.normalization,
            coeffs=args.coeffs,
            overwrite_standard=args.standard,
            overwrite_grey=args.grey,
            overwrite_summary=args.summary,
            cores=args.cores,
            sort=args.sort
        )






if __name__ == "__main__":
    n_meth = 2 * len(NORM_METHODS)
    int2norm = dict(enumerate(NORM_METHODS + [norm + '+indulge' for norm in NORM_METHODS]))
    position2params = [(how, indulge) for indulge, how in product((False, True), NORM_METHODS)]
    parser = argparse.ArgumentParser(
        description="Create Debussy data."
    )
    parser.add_argument(
        "data",
        metavar="DATA_DIR",
        type=check_and_create,
        help="Directory where the NumPy arrays will be stored.",
    )
    parser.add_argument(
        "-w",
        "--wavescapes",
        metavar="WAVESCAPE_DIR",
        type=check_and_create,
        help="If you don't pass this argument, no wavescapes will be created."
    )
    parser.add_argument(
        "-n",
        "--normalization",
        nargs="+",
        metavar="METHOD",
        type=int,
        help=f"By default, all {n_meth} normalization methods are being applied. Pass one or "
             f"several numbers to use only some of them: {int2norm}."
    )
    parser.add_argument(
        "--coeffs",
        nargs="+",
        type=int,
        help="By default, wavescapes for all coefficients are created (if -w is set). Pass one "
             "or several numbers within [1, 6] to create only these."
    )
    parser.add_argument(
        "-c",
        "--cores",
        default=0,
        type=int,
        metavar="N",
        help="Defaults to 0, meaning that all available CPU cores are used in parallel to speed up "
             "the computation. Pass the desired number of cores or a negative number to deactivate."
    )
    parser.add_argument(
        "-s",
        "--sort",
        action='store_true',
        help="This flag influences the processing order by sorting the data matrices from shortest "
             "to longest"
    )
    parser.add_argument(
        "-r",
        "--repo",
        metavar="DIR",
        type=check_dir,
        default='..', #os.getcwd()
        help="Local clone of the debussy repository. Defaults to current working directory.",
    )

    overwriting_group = parser.add_argument_group(title='What kind of existing data to overwrite?')
    overwriting_group.add_argument(
        "--all",
        action='store_true',
        help="Set this flag to create all data from scratch. This amounts to all flags below."
    )
    overwriting_group.add_argument(
        "--magphase",
        action="store_true",
        help="Set this flag to re-compute normalized magnitude-phase matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--correlations",
        action="store_true",
        help="Set this flag to re-compute correlation matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--matrices",
        action="store_true",
        help="Short for --magphase --correlations"
    )
    overwriting_group.add_argument(
        "--standard",
        action="store_true",
        help="Set this flag to re-compute correlation matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--grey",
        action="store_true",
        help="Set this flag to re-compute correlation matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--summary",
        action="store_true",
        help="Set this flag to re-compute correlation matrices even if they exist "
             "already in the data directory."
    )
    overwriting_group.add_argument(
        "--ws",
        action="store_true",
        help="Short for --standard --grey --summary. In other words, re-create all wavescapes but "
             "not the matrices."
    )

    args = parser.parse_args()
    # normalization param
    if args.normalization is None:
        args.normalization = position2params
    else:
        params = []
        for i in args.normalization:
            assert 0 <= i < n_meth, f"Arguments for -n need to be between 0 and {n_meth}, not {i}."
            params.append(position2params[i])
        args.normalization = params
    # multiprocessing param
    available_cpus = mp.cpu_count()
    if args.cores  == 0:
        args.cores = available_cpus
    elif args.cores > available_cpus:
        print(f"{args.cores} CPUs not available, setting the number down to {available_cpus}.")
        args.cores = available_cpus
    elif args.cores < 0:
        args.cores = 0 # deactivates multiprocessing
    # coefficients param
    if args.coeffs is not None:
        for i in args.coeffs:
            assert 0 < i < 7, f"Arguments for --coeff need to be within [1,6], not {i}"
    # overwriting params
    if args.all:
        args.matrices = True
        args.ws = True
    if args.matrices:
        args.magphase = True
        args.correlations = True
    if args.ws:
        args.standard = True
        args.grey = True
        args.summary = True
    main(args)


