import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import traceback
from modules.utils import utils
from modules.GazeMM.irisRecognitionTrain import irisRecognition

DATA_1 = "7_CASIA Iris Image Database (version 1.0)"
DATA_2 = "2_CASIA-Iris-Thousand"
DATA_3 = "0_CASIA_Iris_Interval"
DATA_4 = "5_MMU-v1"
DATA_5 = "3_CASIA-Iris-Lamp"
DATA_6 = "4_PolyU_Iris_DB"
DATA_7 = "8_IIITD_Contact_Lens_Iris_DB_Cogent Scanner"
DATA_8 = "9_IIITD_Contact_Lens_Iris_DB_Vista Scanner"

DATA_9 = "10_eye_movement_dataset_conversion"
DATA_10 = "11_openEDS"
DATA_11 = "14_CASIA-Iris-Degradation 1.0"
DATA_12 = "15_CASIA-Iris-Africa 1.0"

CV_FOLDS = 2
CV_REPEATS = 3
CV_SEED = 42

MANIFOLD_COMPONENTS = 3
MANIFOLD_MIN_IMGS = 3

FOCUS_GATE_C = 2.0

LOAD_DATASETS = [DATA_1, DATA_2, DATA_3, DATA_4, DATA_5, DATA_6,
                 DATA_7, DATA_8, DATA_9, DATA_10, DATA_11, DATA_12]


if __name__ == "__main__":
    util = utils()

    irisRec = irisRecognition()
    irisRec.cv_folds = CV_FOLDS
    irisRec.cv_repeats = CV_REPEATS
    irisRec.cv_seed = CV_SEED
    irisRec.manifold_components = MANIFOLD_COMPONENTS
    irisRec.manifold_min_imgs = MANIFOLD_MIN_IMGS
    irisRec.focus_gate_c = FOCUS_GATE_C

    summary = {}
    summary_fte = {}
    for ds in LOAD_DATASETS:
        ds_path = "./data/" + ds
        if not os.path.isdir(ds_path):
            print(f"[{ds}] data folder not found, skipping")
            continue
        print("\n" + "=" * 70 + f"\nDataset: {ds}\n" + "=" * 70)
        summary[ds] = None
        summary_fte[ds] = float("nan")
        try:
            filenames = util.list_image_paths(ds_path)
            print(f"[{ds}] {len(filenames)} images (left and right eyes)")
            irisRec.keep_aspect = (ds == DATA_10)
            irisRec.fixed_splits = util.make_splits(filenames, folds=CV_FOLDS,
                                                    repeats=CV_REPEATS, seed=CV_SEED)
            summary[ds] = irisRec.run(filenames)
            summary_fte[ds] = irisRec.fte_pct
        except Exception as e:
            print(f"[{ds}] failed: {e}")
            traceback.print_exc()

    if summary:
        W = 40 + 16 + 15 + 15 + 10
        print("\n" + "=" * W)
        print(f"Summary: subject-disjoint {CV_REPEATS}x{CV_FOLDS}-fold cross-validation (open-set). "
              f"Values are mean ± std over folds (population std).")
        print("FTE% = percentage of images discarded by the focus-quality gate (ISO/IEC 19795).")
        print("-" * W)
        print(f"{'dataset':<40}{'FRR@1e-3%':>16}{'EER%':>15}{'Rank-1%':>15}{'FTE%':>10}")
        print("-" * W)
        for ds, res in summary.items():
            _fp = summary_fte[ds]
            _fs_txt = "N/A" if _fp != _fp else f"{_fp:.2f}"
            if res is None:
                print(f"{ds:<40}{'N/A':>16}{'N/A':>15}{'N/A':>15}{_fs_txt:>10}")
            else:
                fm, fs, em, es, rm, rs = res
                print(f"{ds:<40}"
                      f"{f'{fm:.2f}±{fs:.2f}':>16}"
                      f"{f'{em:.2f}±{es:.2f}':>15}"
                      f"{f'{rm:.2f}±{rs:.2f}':>15}"
                      f"{_fs_txt:>10}")
        print("=" * W)
