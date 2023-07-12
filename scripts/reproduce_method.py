import pandas as pd
from scripts.do_inference import do_inference
from scripts.get_checkpoint import get_checkpoint
from argparse import ArgumentParser
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('--method_names', nargs='+', required=True)
    parser.add_argument('--dataset', default='ADE20K')
    parser.add_argument('--resolution', nargs=2, default=[512, 512])
    parser.add_argument('--only_download', action='store_true')
    parser.add_argument('--a100', action='store_true')
    parser.add_argument('--submit_slurm_jobs', action='store_true')
    args = parser.parse_args()
    if type(args.resolution) in [list, tuple]:
        args.resolution = str(tuple(int(x) for x in args.resolution)).replace(', ', ',')
    print(f'{args=}')
    reproduce_method(
        methods=args.method_names,
        dataset_name=args.dataset,
        resolution=args.resolution,
        only_download=args.only_download,
        submit_slurm_jobs=args.submit_slurm_jobs,
        a100=args.a100,
    )


def reproduce_method(methods, dataset_name, resolution, only_download=False, submit_slurm_jobs=False, a100=False):
    for method in methods:
        input_shape = (3, *eval(resolution))
        df = pd.read_csv(f'tables/from_mmseg/{dataset_name}/{resolution}.csv')
        df = df[df.method==method]
        for config in df.config:
            print(f"Start Script for config: {config}")
            if only_download:
                get_checkpoint(config_path=config)
            elif submit_slurm_jobs:
                if a100:
                    os.system(f'sbatch scripts/do_inference_a100.sh {config}')
                else:
                    os.system(f'sbatch scripts/do_inference.sh {config}')
            else:
                do_inference(config, input_shape=input_shape)


if __name__ == '__main__':
    main()
