import pandas as pd
from scripts.do_inference import do_inference
from scripts.get_checkpoint import get_checkpoint
from argparse import ArgumentParser
import os
import os.path as osp


def main():
    parser = ArgumentParser()
    parser.add_argument('--method_names', nargs='+', required=True)
    parser.add_argument('--dataset', default='ADE20K')
    parser.add_argument('--resolution', nargs=2, default=[512, 512])
    parser.add_argument('--only_download', action='store_true')
    parser.add_argument('--a100', action='store_true')
    parser.add_argument('--submit_slurm_jobs', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--work-dir-postfix', default='')
    parser.add_argument('--batch-size', default=None)
    args = parser.parse_args()
    if type(args.resolution) in [list, tuple]:
        args.resolution = str(tuple(int(x) for x in args.resolution)).replace(', ', ',')
    print(f'{args=}')
    if args.train:
        train_method(
            methods=args.method_names,
            dataset_name=args.dataset,
            resolution=args.resolution,
            a100=args.a100,
            work_dir_postfix=args.work_dir_postfix,
            batch_size=args.batch_size
        )
    else:
        reproduce_method(
            methods=args.method_names,
            dataset_name=args.dataset,
            resolution=args.resolution,
            only_download=args.only_download,
            submit_slurm_jobs=args.submit_slurm_jobs,
            a100=args.a100,
        )

def train_method(methods, dataset_name, resolution, a100=False, work_dir_postfix="",batch_size=None):
    batch_size_str = ""
    if batch_size is not None:
        batch_size_str = f" train_dataloader.batch_size={batch_size}"

    for method in methods:
        df = pd.read_csv(f'tables/from_mmseg/{dataset_name}/{resolution}.csv')
        df = df[df.method==method]
        for config in df.config:
            work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
            work_dir = (work_dir + "_" + work_dir_postfix) if work_dir_postfix != "" else work_dir
            print(f"Start Training for config: {config}")
            if a100:
                os.system(f'sbatch tools/slurm_train_ifn_resume_a100.sh {config} --work-dir {work_dir}{batch_size_str}')
            else:
                os.system(f'sbatch tools/slurm_train_ifn_resume.sh {config} --work-dir {work_dir}{batch_size_str}')
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
