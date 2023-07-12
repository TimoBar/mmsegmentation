import os.path
import matplotlib.pyplot as plt
import pandas as pd
from dotdict import dotdict
from glob import glob


def main():
    plt.style.use('seaborn')
    csv_dir = 'tables/from_mmseg'
    dataset_name, resolution = 'ADE20K', '(512,512)'
    # dataset_name, resolution = 'Cityscapes', '(512,1024)'
    df = pd.read_csv(f'{csv_dir}/{dataset_name}/{resolution}.csv')
    df['config_name'] = df.config.apply(lambda x: os.path.basename(x) if x and isinstance(x, str) else x)
    df = find_best_methods(df)
    a100 = False
    # a100 = True
    if a100:
        gpu = 'a100'
        df_reproduced = read_reproduced_results(suffix='_a100')
    else:
        gpu = '1080ti'
        df_reproduced = read_reproduced_results()
    if df_reproduced is None:
        df_merged = df
    else:
        df_reproduced.columns = [col + '_reproduced' if not col=='config_name' else col for col in df_reproduced.columns]
        df_merged = pd.merge(df, df_reproduced, how='outer', on='config_name')
    plot_miou_over_fps(df_merged, plot_name=f'{dataset_name}, resolution={resolution}, gpu={gpu}')




def plot_miou_over_fps(df_mmseg, plot_name):
    x, x_reproduced = 'fps', 'fps_reproduced'
    # x, x_reproduced = 'flops_reproduced', 'flops_reproduced'
    # x, x_reproduced = 'params_reproduced', 'params_reproduced'

    # draw_reproduced = False
    draw_reproduced = True
    # draw_reported = False
    draw_reported = True
    fig, ax = plt.subplots(dpi=200, figsize=(9, 5))

    fig.suptitle(plot_name)
    # ax2 = ax.twiny()
    # ax2.set_xlabel('fps (reproduced)')
    for method, sub_df in df_mmseg.groupby('method'):
        is_reproduced = 'fps_reproduced' in sub_df.columns and len(sub_df.fps_reproduced.dropna()) > 0
        is_reported = 'fps' in sub_df.columns and len(sub_df.fps.dropna()) > 0
        is_intesting = any([
            # method in '''
            #     segformer
            #     segnext
            # '''.split(),
            # any(sub_df.is_best),
            is_reproduced,
        ])


        label = method if is_intesting else None
        plot_kw = dotdict(marker='X', ms=4)
        if is_reproduced:
            plot_kw.zorder = 10
            plot_kw.lw = 2
            plot_kw.ms = 6
        else:
            plot_kw.lw = 1.5

        if is_intesting:
            plot_kw.alpha=0.6
        else:
            plot_kw.color = 'gray'
            plot_kw.alpha=0.3
        if is_reported and draw_reported:
            _label = None if draw_reproduced else label
            plot_kw.color = ax.plot(sub_df[x], sub_df.mIoU, label=_label, **plot_kw)[0]._color
        if is_reproduced and is_intesting and draw_reproduced:
            plot_kw.marker='o'
            plot_kw.ms=8
            ms = sub_df.params_reproduced.apply(lambda x: x / 100_000)
            ms = sub_df.flops_reproduced.apply(lambda x: x / 1000_000_000)
            ax.plot(sub_df[x_reproduced], sub_df.mIoU_reproduced, **{**plot_kw, 'alpha': 1, 'lw': 3, 'label': label})
            # ax.scatter(sub_df.fps, sub_df.mIoU, s=20, marker='X')
            for i, row in sub_df.dropna(axis=0).iterrows():
                plt.plot(
                    [row.fps, row.fps_reproduced],
                    [row.mIoU, row.mIoU_reproduced],
                    color=plot_kw.get('color', 'gray'),
                    ls='dotted',
                    alpha=plot_kw.alpha,
                    zorder=5
                )
    ax.set_xlabel(x)
    ax.set_ylabel('mIoU')
    plt.legend()
    plt.tight_layout()
    # ax.set_ylim(bottom=20)
    # ax.set_xlim(right=250e9)
    plt.show()


def read_reproduced_results(suffix=''):
    dfs = []
    filenames = glob(f'tables/reproduced{suffix}/*.csv')
    if not filenames:
        return None
    for filename in filenames:
        config_name = os.path.basename(filename)[:-4]
        df = pd.read_csv(filename)
        df['config_name'] = config_name
        dfs.append(df)
    reproduced_df = pd.concat(dfs)
    return reproduced_df

def find_best_methods(df):
    df['is_best'] = False
    best_mIoU = 0
    for i, mIoU in enumerate(df.mIoU):
        if mIoU > best_mIoU:
            df['is_best'].iloc[i] = True
            best_mIoU = mIoU
    return df


if __name__ == '__main__':
    main()
