import yaml
from glob import glob
import pandas as pd
from dotdict import dotdict
import os
import re

def main():
    csv_dir = 'tables/from_mmseg'
    groups = create_csvs(csv_dir)


def create_csvs(csv_dir):
    yaml_files = glob('configs/**/*.yaml')
    res_list = []
    for f in list(yaml_files):
        res_list += extract_results(f)
    df = pd.DataFrame([dict(r) for r in res_list])
    df = df.sort_values(by='fps', ascending=False)
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(f'{csv_dir}/all.csv', index=False)
    groups = {k: v for k, v in df.groupby(['dataset', 'resolution'])}
    for (ds, res), group in groups.items():
        ds_dir = f'{csv_dir}/{ds}'
        print(ds_dir)
        os.makedirs(ds_dir, exist_ok=True)
        group.to_csv(f'{csv_dir}/{ds}/{res}.csv', index=False)
    return groups


def parse_resolution_from_name(config_name):
    res = re.search("[a-zA-Z_-][0-9]*x[0-9]*[.-_]", config_name)[0][1:-1]
    return f"({res.replace('x', ',')})"

def extract_results(yaml_file):
    res_list = []
    with open(yaml_file, 'r') as f:
        file_content = yaml.safe_load(f)
    dir_name = os.path.basename(os.path.dirname(yaml_file))
    models = file_content['Models']
    for model in models:
        try:
            d_model = dotdict()
            d_model.name = model['In Collection']
            d_model.method = dir_name
            mdat = model['Metadata']
            if 'Architecture' in mdat:
                d_model.name += '_' + mdat['Architecture'][0]
            d_model.config = model['Config']
            d_model.weights = model['Weights']
            d_inf_time = mdat.get('inference time (ms/im)', '')
            d_model.fps = ""
            d_model.hardware = mdat.get('Training Resources', "")


            d_model.resolution = parse_resolution_from_name(d_model.config)
            #print(d_model.resolution)

            results = model['Results'] if isinstance(model['Results'], list) else [model['Results']]
            results = [r for r in results if r['Task'] == 'Semantic Segmentation']
            assert len(results) == 1, 'd_inf_time has multiple elements'
            res = results[0]
            #print(res['Metrics'])
            d_res = dotdict(d_model)
            #print(res['Dataset'])
            d_res.dataset = res['Dataset']
            d_res.mIoU = res['Metrics']['mIoU']
            res_list.append(d_res)
        except KeyError as key_error:
            pass
            #print(key_error)
    return res_list



if __name__ == '__main__':
    main()