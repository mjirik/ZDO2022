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

import zdo2022.main
# cd ZDO2022
# python -m pytest

def test_run_random():
    vdd = zdo2022.main.InstrumentTracker()

    # Nastavte si v operačním systém proměnnou prostředí 'ZDO_DATA_PATH' s cestou k datasetu.
    # Pokud není nastavena, využívá se testovací dataset tests/test_dataset
    dataset_path = Path(os.getenv('ZDO_PATH_', default=Path(__file__).parent / 'test_dataset/'))
    # dataset_path = Path(r"H:\biology\orig\zdo_varroa_detection_coco_001")

    print(f'dataset_path = {dataset_path}')
    types = ('*.MP4', '*.mkv')  # the tuple of file types
    files = []
    for file_type in types:
        files.extend(dataset_path.glob(f"./{file_type}"))
    print(f"files={files}")
    cislo_souboru = np.random.randint(0, len(files))
    filename = files[cislo_souboru]

    prediction = vdd.predict(filename)

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
    vdd = zdo2022.main.InstrumentTracker()

    # Nastavte si v operačním systém proměnnou prostředí 'ZDO_DATA_PATH' s cestou k datasetu.
    # Pokud není nastavena, využívá se testovací dataset tests/test_dataset
    dataset_path = os.getenv('ZDO_DATA_PATH_', default=Path(__file__).parent / 'test_dataset/')
    # dataset_path = Path(r"H:\biology\orig\zdo_varroa_detection_coco_001")

    # print(f'dataset_path = {dataset_path}')
    types = ('*.MP4', '*.mkv')  # the tuple of file types
    files = []
    for file_type in types:
        files.extend(dataset_path.glob(f"./{file_type}"))

    f1s = []
    for filename in files:
        prediction = vdd.predict(filename)

        df_eval = check_one_prediction(filename, prediction)
        f1s.append(df_eval)

    df = pd.concat(f1s)
    print(df)
    # assert f1 > 0.55


def check_one_prediction(filename:Path, prediction:dict) -> pd.DataFrame:
    ann_pth = filename.with_suffix(".xml")
    assert ann_pth.exists()
    ground_true_annotation = xml_to_dict(ann_pth)

    df_gt = pd.DataFrame(ground_true_annotation)
    # df_gt.frame_id = pd.to_numeric(df_gt.frame_id)
    # df_gt.object_id = pd.to_numeric(df_gt.object_id)
    df_prediction = pd.DataFrame(prediction)
    # df_prediction.frame_id = pd.to_numeric(df_prediction.frame_id)
    # df_prediction.object_id = pd.to_numeric(df_prediction.object_id)

    df_eval = dist_eval(df_gt, df_prediction)

    return df_eval


def xml_to_dict(pth:Path):

    annotation = {
        "filename": [],  # pth.parts[-1]
        "frame_id": [],
        "object_id": [],
        "x_px": [],  # x pozice obarvených hrotů v pixelech
        "y_px": [],  # y pozice obarvených hrotů v pixelech
        "annotation_timestamp": [],
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

def dist_eval(df_gt, df_algorithm):
    df_anns = df_gt
    fns = []
    dst_max = []
    dst_std = []
    dst_mean = []
    object_ids = []
    debug_per = []
    for fn in df_algorithm.filename.unique():
        df_anns_sel = df_anns[df_anns.filename==fn]
        n_frames = max(np.max(df_anns_sel.frame_id), np.max(df_algorithm.frame_id)) + 1
        n_objects = max(np.max(df_anns_sel.object_id), np.max(df_algorithm.object_id)) + 1

        object_i = 0
        object_j = 0

        df_an_s_fn = interpolate_px(df_anns[(df_anns.filename == fn)], n_frames=n_frames, n_objects=n_objects)
        df_al_s_fn = interpolate_px(df_algorithm[(df_algorithm.filename == fn)], n_frames=n_frames, n_objects=n_objects)
        for object_i in range(n_objects):
            dist_mean = None
            dist_min = 0
            dist_std = 0
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
    #                     print(f"dist_mean={dist_mean} in init")
                    else:
                        if mn < dist_mean:
                            dist_mean = np.mean(per_fram_dist)
                            dist_std = np.std(per_fram_dist)
                            dist_max = np.max(per_fram_dist)
    #                         print(f"dist_mean={dist_mean} in else")
            fns.append(fn)
            dst_max.append(dist_max)
            dst_std.append(dist_std)
            dst_mean.append(dist_mean)
    #         print(f"dist_mean={dist_mean}")
            object_ids.append(object_i)

    df = pd.DataFrame(dict(filename=fns, object_id=object_ids, dist_max=dst_max, dist_std=dst_std, dist_mean=dst_mean))
    return df
