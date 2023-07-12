import pandas as pd
import os
import wget


def detect_filename(url=None, out=None, headers=None, default="download.wget"):
    """Return filename for saving file. If no filename is detected from output
    argument, url or headers, return default (download.wget)
    """
    name = default
    if out:
        name = out
    elif url:
        name = wget.filename_from_url(url)
    elif headers:
        name = wget.filename_from_headers(headers)
    return name


def fix_wget():
    wget.detect_filename = detect_filename


def get_checkpoint(config_path):
    fix_wget()
    ckpt_file = config_path.replace('configs/', 'downloaded_ckpts/') + '.pth'
    if not os.path.exists(ckpt_file):
        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
        df = pd.read_csv('tables/from_mmseg/all.csv')
        print(config_path)
        print(df[df.config==config_path])
        ckpt_url = df[df.config==config_path].weights.iloc[0]
        ckpt_url = ckpt_url.replace('https:', 'http:')
        print(f'downloading weights from {ckpt_url}')
        wget.download(ckpt_url, ckpt_file)
    return ckpt_file


if __name__ == '__main__':
    ckpt_file = get_checkpoint('configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py')
    print(ckpt_file)