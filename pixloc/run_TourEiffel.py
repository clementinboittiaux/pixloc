import pickle

from . import set_logging_debug, logger
from .localization import RetrievalLocalizer, PoseLocalizer
from .utils.data import Paths, create_argparser, parse_paths, parse_conf
from .utils.io import write_pose_results
from .utils.eval import evaluate

default_paths = Paths(
    query_images='{suffix}/images',
    reference_images='{suffix}/images',
    reference_sfm='{suffix}/sfm_superpoint+superglue/',
    query_list='{suffix}/query_list_with_intrinsics.txt',
    retrieval_pairs='{suffix}/pairs-query-netvlad10.txt',
    ground_truth='{suffix}/',
    results='pixloc_TourEiffel_{suffix}{from_poses}.txt',
    hloc_logs='{suffix}/results.txt_logs.pkl'
)

experiment = 'pixloc_megadepth'

default_confs = {
    'from_retrieval': {
        'experiment': experiment,
        'features': {},
        'optimizer': {
            'num_iters': 150,
            'pad': 1,
        },
        'refinement': {
            'num_dbs': 3,
            'multiscale': [4, 1],
            'point_selection': 'all',
            'normalize_descriptors': True,
            'average_observations': False,
            'do_pose_approximation': True,
        },
    },
    'from_poses': {
        'experiment': experiment,
        'features': {'preprocessing': {'resize': 1600}},
        'optimizer': {
            'num_iters': 50,
            'pad': 1,
        },
        'refinement': {
            'num_dbs': 5,
            'min_points_opt': 100,
            'point_selection': 'inliers',
            'normalize_descriptors': True,
            'average_observations': True,
            'layer_indices': [0, 1],
        },
    },
}


def main():
    parser = create_argparser('TourEiffel')
    parser.add_argument('--suffix', required=True)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    set_logging_debug(args.verbose)
    paths = parse_paths(args, default_paths)
    paths.dataset /= '{suffix}'
    paths.dumps /= '{suffix}'
    conf = parse_conf(args, default_confs)

    logger.info('Working on dataset %s.', f'TourEiffel_{args.suffix}')
    paths = paths.interpolate(suffix=args.suffix, from_poses=f'{"_from_poses" if args.from_poses else ""}')
    if args.eval_only and paths.results.exists():
        poses = paths.results
    else:
        if args.from_poses:
            localizer = PoseLocalizer(paths, conf)
        else:
            localizer = RetrievalLocalizer(paths, conf)
        poses, logs = localizer.run_batched(skip=args.skip)
        write_pose_results(poses, paths.results,
                           prepend_camera_name=False)
        with open(f'{paths.results}_logs.pkl', 'wb') as f:
            pickle.dump(logs, f)

    logger.info('Evaluate dataset %s: %s', f'TourEiffel_{args.suffix}', paths.results)
    evaluate(paths.ground_truth / 'empty_all', poses,
             paths.ground_truth / 'list_query.txt',
             only_localized=(args.skip is not None and args.skip > 1))


if __name__ == '__main__':
    main()
