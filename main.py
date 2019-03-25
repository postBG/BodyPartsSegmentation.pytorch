import pprint as pp

from misc import create_experiment_export_folder, export_experiments_config_as_json, fix_random_seed_as, set_up_gpu
from options import args as parsed_args


def main(args):
    pass


def setup_experiments(args):
    set_up_gpu(args)
    fix_random_seed_as(args.random_seed)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    pp.pprint(vars(args), width=1)
    return export_root


if __name__ == "__main__":
    main(parsed_args)
