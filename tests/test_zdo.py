import matplotlib.pyplot as plt
import pytest
import os
import skimage.io
from typing import Optional
from skimage.draw import polygon
import glob
import numpy as np
from pathlib import Path
import sklearn.metrics
import pandas as pd
from pathlib import Path
import lxml
from lxml import etree
import argparse
import sys
import logging

# cd ZDO2022
# python -m pytest


# Nastavte si v operačním systém proměnnou prostředí 'ZDO_DATA_PATH' s cestou k datasetu.
# Pokud není nastavena, využívá se testovací dataset tests/test_dataset
dataset_path = Path(os.getenv('ZDO_PATH_', default=Path(__file__).parent / 'test_dataset/'))
# dataset_path = Path(r"H:\biology\orig\zdo_varroa_detection_coco_001")
# dataset_path = Path(r"G:\Můj disk\data\biomedical\orig\pigleg_surgery\ZDO2022")
show = False

def test_run_random():
    import zdo2022.main
    vdd = zdo2022.main.InstrumentTracker()

    print(f'dataset_path = {dataset_path}')
    types = ('*.MP4', '*.mkv')  # the tuple of file types
    files = []
    for file_type in types:
        files.extend(dataset_path.glob(f"./{file_type}"))
    print(f"files={files}")
    cislo_souboru = np.random.randint(0, len(files))
    filename = files[cislo_souboru]

    prediction = vdd.predict(str(filename))

    # im = skimage.io.imread(filename)
    # imgs = np.expand_dims(im, axis=0)
    # # print(f"imgs.shape={imgs.shape}")
    # prediction = vdd.predict(imgs)

    # assert prediction.shape[0] == imgs.shape[0]

    # Toto se bude spouštět všude mimo GitHub
    if not os.getenv('CI'):
        import seaborn as sns
        import matplotlib.pyplot as plt
        df_interpolated = interpolate_px(pd.DataFrame(prediction))
        sns.lineplot(data=df_interpolated, x="x_px", y="y_px", hue="object_id", sort=False)
        # plt.imshow(prediction[0])
        plt.show()

def test_run_all():
    import zdo2022.main
    vdd = zdo2022.main.InstrumentTracker()


    # print(f'dataset_path = {dataset_path}')
    types = ('*.MP4', '*.mkv')  # the tuple of file types
    files = []
    for file_type in types:
        files.extend(dataset_path.glob(f"./{file_type}"))

    f1s = []
    for filename in files:
        ann_pth = filename.with_suffix(".xml")
        if filename.exists() and ann_pth.exists():
            prediction = vdd.predict(str(filename))


            df_eval = check_one_prediction(filename, prediction)
            f1s.append(df_eval)

    df = pd.concat(f1s)
    if show():
        plt.show()
    # print(df)
    return df
    # assert f1 > 0.55


def check_one_prediction(filename:Path, prediction:dict) -> pd.DataFrame:
    import zdo2022
    zdo_pth = Path(zdo2022.__file__).parent.parent
    name = zdo_pth.name
    ann_pth = filename.with_suffix(".xml")
    assert ann_pth.exists()
    ground_true_annotation = xml_to_dict(ann_pth)

    df_gt = pd.DataFrame(ground_true_annotation)
    label_map = {
        "needle holder" : 0,
        "scissors": 1,
        "tweezers": 2
    }
    df_gt["object_id"] = df_gt.label.map(lambda oid: label_map[oid])
    # df_gt.frame_id = pd.to_numeric(df_gt.frame_id)
    # df_gt.object_id = pd.to_numeric(df_gt.object_id)
    df_prediction = pd.DataFrame(prediction)
    df_prediction = df_prediction[df_prediction.x_px >= 0] # filter out negative values
    # df_prediction.frame_id = pd.to_numeric(df_prediction.frame_id)
    # df_prediction.object_id = pd.to_numeric(df_prediction.object_id)

    df_prediction.to_csv(f"{name}_{filename.name}_prediction.csv")
    # print(df_prediction)
    #
    # print(df_gt)
    import seaborn as sns
    # import ipdb
    # ipdb.set_trace()
    df_gt["ground_true"] = True
    df_prediction["ground_true"] = False
    dfall = pd.concat([df_gt, df_prediction], axis=0)
    dfall = dfall.reset_index()

    df_eval = dist_eval(df_gt, df_prediction)

    df_eval.to_csv(f"{name}_{filename.name}_eval.csv")
    dfall.to_csv(f"{name}_{filename.name}_all.csv")
    plt.figure()
    sns.lineplot(data=dfall[dfall.object_id == 0], x="x_px", y="y_px", hue="ground_true", sort=False)
    plt.savefig(f"{name}_{filename.name}_trajectory.pdf")

    return df_eval


def xml_to_dict(pth:Path):

    annotation = {
        "filename": [],  # pth.parts[-1]
        "frame_id": [],
        "object_id": [],
        "x_px": [],  # x pozice obarvených hrotů v pixelech
        "y_px": [],  # y pozice obarvených hrotů v pixelech
        "annotation_timestamp": [],
        "label": [],
    }

    tree = etree.parse(pth)

    updated = tree.xpath("//updated")[0].text  # date of last change in CVAT

    for track in tree.xpath('track'):
        for point in track.xpath("points"):
            pts = point.get("points").split(",")
            x, y = pts
            # annotation["filename"].append(str(pth))
            annotation["filename"].append(tree.xpath("//source")[0].text)
            annotation["object_id"].append(int(track.get("id")))
            annotation["label"].append(track.get("label"))
            annotation["x_px"].append(float(x))
            annotation["y_px"].append(float(y))
            annotation["frame_id"].append(int(point.get("frame")))
            annotation["annotation_timestamp"].append(updated)

    return annotation


def interpolate_px(df_algorithm:pd.DataFrame, n_frames:Optional[int]=None, n_objects:Optional[int]=None, order:int=1) -> pd.DataFrame:
    if n_frames:
        l = n_frames
    else:
        l = np.max(df_algorithm.frame_id) + 1
    if not n_objects:
        n_objects = np.max(df_algorithm.object_id)
    df_interpolated = pd.DataFrame()

    for fn_i in df_algorithm.filename.unique():
        for object_id_i in range(0, n_objects + 1):
            df_append = pd.DataFrame(dict(
                filename=[fn_i] * l,
                frame_id=list(range(l)),
                object_id=[object_id_i] * l,
                x_px=[np.nan] * l,
                y_px=[np.nan] * l,
            ))

            one_to_append = df_algorithm[
                (df_algorithm.object_id==object_id_i) & (df_algorithm.filename==fn_i)
            ].append(df_append).sort_values(
                by=["frame_id", "y_px"]
            ).reset_index(drop=True).drop_duplicates(
                "frame_id"
            ).interpolate( # this is not putting data in front or to the back
                    method='polynomial', order=order,
    #                 limit_area=None,limit_direction="forward"
            ).interpolate( # this will fill the data
                limit_area=None,limit_direction="backward"
            ).interpolate(
                limit_area=None,limit_direction="forward"
            )

            df_interpolated = df_interpolated.append(
                one_to_append, sort=False
            )
    df_interpolated = df_interpolated.sort_values(by=["filename" ,  "object_id", "frame_id", ]).reset_index(drop=True)
    return df_interpolated

def dist_eval(df_gt, df_algorithm, interpolate_algorithm=False):
    df_anns = df_gt
    df_anns.filename = df_anns.filename.map(lambda pth: str(Path(pth).name))
    df_algorithm.filename = df_algorithm.filename.map(lambda pth: str(Path(pth).name))

    dict_all = {
        "filename": [],
        "x_px": [],
        "y_px": [],
        "x_px_gt": [],
        "y_px_gt": [],
        "object_id": [],
        "frame_id": [],
        "diff": []
    }
    fns = []
    dst_max = []
    dst_std = []
    dst_mean = []
    dst_median = []
    object_ids = []
    debug_per = []
    for fn in df_algorithm.filename.unique():
        df_anns_sel = df_anns[df_anns.filename==fn]
        n_frames = max(np.max(df_anns_sel.frame_id), np.max(df_algorithm.frame_id)) + 1
        n_objects = max(np.max(df_anns_sel.object_id), np.max(df_algorithm.object_id)) + 1

        object_i = 0
        object_j = 0

        # import ipdb
        # ipdb.set_trace()
        df_an_s_fn = interpolate_px(df_anns[(df_anns.filename == fn)], n_frames=n_frames, n_objects=n_objects)
        if interpolate_algorithm:
            df_al_s_fn = interpolate_px(df_algorithm[(df_algorithm.filename == fn)], n_frames=n_frames, n_objects=n_objects)
        else:
            df_al_s_fn = df_algorithm[(df_algorithm.filename == fn)]
        for object_i in range(n_objects):
            dist_mean = None
            dist_min = 0
            dist_std = 0
            dist_median = 0
            for object_j in range(n_objects):

                df_an_s = df_an_s_fn[(df_an_s_fn.object_id==object_j)]
                df_al_s = df_al_s_fn[(df_al_s_fn.object_id==object_i)]

                per_fram_dist = ((df_an_s.x_px - df_al_s.x_px)**2 + (df_an_s.y_px - df_al_s.y_px)**2)**0.5
                debug_per.append(per_fram_dist)

                mn = np.mean(per_fram_dist)
#                 print(f"mn={mn} , {type(mn)} {mn is nan}")

                if mn is not np.nan:
                    if dist_mean is None:
                        dist_mean = np.mean(per_fram_dist)
                        dist_std = np.std(per_fram_dist)
                        dist_max = np.max(per_fram_dist)
                        dist_median = np.median(per_fram_dist)
    #                     print(f"dist_mean={dist_mean} in init")
                    else:
                        if mn < dist_mean:
                            dist_mean = np.mean(per_fram_dist)
                            dist_std = np.std(per_fram_dist)
                            dist_max = np.max(per_fram_dist)
                            dist_median = np.median(per_fram_dist)
    #                         print(f"dist_mean={dist_mean} in else")
            fns.append(fn)
            dst_max.append(dist_max)
            dst_std.append(dist_std)
            dst_mean.append(dist_mean)
    #         print(f"dist_mean={dist_mean}")
            object_ids.append(object_i)
            dst_median.append(dist_median)

    df = pd.DataFrame(dict(filename=fns, object_id=object_ids, dist_max=dst_max, dist_std=dst_std, dist_mean=dst_mean, dist_median=dst_median))
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="display a square of a given number",
                        type=str)
    parser.add_argument("--dataset-path", help="path to dataset",
                        default=str(os.getenv('ZDO_PATH_', default=Path(__file__).parent / 'test_dataset/')),
                        type=str)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.path).absolute()))
    dataset_path = Path(args.dataset_path)

    import zdo2022.main

    show = True
    df = test_run_all()
    print(df)
    # df.to_csv()