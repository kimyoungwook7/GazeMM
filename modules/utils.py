import glob
import os
import numpy as np

EXTENSIONS = ["bmp","png","gif","jpg","jpeg","tiff","tif"]


class utils(object):

    def get_subject_id(self, path):
        parts = os.path.normpath(path).split(os.sep)
        parent = parts[-2] if len(parts) >= 2 else ''
        if parent.upper() in ('L', 'R'):
            person = parts[-3] if len(parts) >= 3 else parent
            return f"{person}__{parent.upper()}"
        return parent


    def make_splits(self, filenames, folds=2, repeats=1, seed=42):
        import random as _rnd
        from collections import defaultdict
        groups = defaultdict(list)
        for f in filenames:
            groups[self.get_subject_id(f)].append(f)
        classes_all = sorted(groups.keys())
        k = int(folds)
        reps = int(repeats)

        out = []
        for rep in range(reps):
            classes = list(classes_all)
            _rnd.Random(seed + rep).shuffle(classes)
            for fi in range(k):
                te = set(classes[fi::k])
                tr_fns = [f for f in filenames if self.get_subject_id(f) not in te]
                te_fns = [f for f in filenames if self.get_subject_id(f) in te]
                if len(te_fns) < 2 or len(tr_fns) < 2:
                    continue
                out.append((f"{rep+1}/{reps}·{k}-fold {fi+1}/{k}", tr_fns, te_fns))
        return out

    def list_image_paths(self, data_dir):
        filenames = []
        folders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        for folder in folders:
            for ext in EXTENSIONS:
                filenames.extend(glob.glob(os.path.join(folder, "**", "*." + ext), recursive=True))
        return filenames


    def cal_FRR_at_FAR_and_EER(self, dist_matrix, N, ids):
        _cnt = np.bincount(ids)
        _n_gen = int((_cnt.astype(np.int64) * (_cnt - 1) // 2).sum())
        _n_imp = int(N * (N - 1) // 2 - _n_gen)
        impostor_scores = np.empty(_n_imp, dtype=dist_matrix.dtype)
        genuine_scores = np.empty(_n_gen, dtype=dist_matrix.dtype)

        _blk_rows = max(1, int(2 ** 24 // max(1, N)))
        _cols = np.arange(N)
        _oi = _og = 0
        for _r0 in range(0, N, _blk_rows):
            _r1 = min(N, _r0 + _blk_rows)
            _row = dist_matrix[_r0:_r1]
            _upper = _cols[None, :] > np.arange(_r0, _r1)[:, None]
            _same = (ids[_r0:_r1, None] == ids[None, :])
            _vi = _row[_upper & ~_same]
            impostor_scores[_oi:_oi + _vi.size] = _vi
            _oi += _vi.size
            _vg = _row[_upper & _same]
            genuine_scores[_og:_og + _vg.size] = _vg
            _og += _vg.size
            del _row, _upper, _same, _vi, _vg
        assert _oi == _n_imp and _og == _n_gen, \
            f"pair count mismatch: impostor {_oi}/{_n_imp}, genuine {_og}/{_n_gen}"

        if len(impostor_scores) == 0 or len(genuine_scores) == 0:
            print(f"[evaluate] Metrics are undefined: {len(impostor_scores)} impostor pairs and "
                  f"{len(genuine_scores)} genuine pairs (N={N}); returning NaN.")
            return float("nan"), float("nan")

        impostor_scores.sort()
        genuine_scores.sort()

        target_far = 1e-3
        thresholds = np.linspace(dist_matrix.min(), dist_matrix.max(), 10000)

        far_list = np.searchsorted(impostor_scores, thresholds, side='left') / len(impostor_scores)
        frr_list = 1.0 - np.searchsorted(genuine_scores, thresholds, side='left') / len(genuine_scores)

        d_eer = np.abs(far_list - frr_list)
        i_eer = int(np.argmin(d_eer))
        best_eer = float((far_list[i_eer] + frr_list[i_eer]) / 2)

        d_far = np.abs(far_list - target_far)
        i_far = len(d_far) - 1 - int(np.argmin(d_far[::-1]))
        best_frr = float(frr_list[i_far])

        return best_frr, best_eer


    def cal_Rank_1(self, dist_matrix, N, ids):
        n_sub = int(len(np.unique(ids)))
        n_multi = int((np.bincount(ids) >= 2).sum())
        if n_sub < 2 or n_multi == 0:
            print(f"[evaluate] Rank-1 is undefined: {n_sub} identities, {n_multi} of them with at least "
                  f"two images; returning NaN.")
            return float("nan")

        pred_idx = np.empty(N, dtype=np.intp)
        _blk_rows = max(1, int(2 ** 24 // max(1, N)))
        for _r0 in range(0, N, _blk_rows):
            _r1 = min(N, _r0 + _blk_rows)
            _blk = np.array(dist_matrix[_r0:_r1], copy=True)
            _blk[np.arange(_r1 - _r0), np.arange(_r0, _r1)] = np.inf
            pred_idx[_r0:_r1] = np.argmin(_blk, axis=1)
            del _blk
        correct = (ids == ids[pred_idx])

        mated = np.bincount(ids)[ids] >= 2
        return float(np.mean(correct[mated]))


    def evaluate(self, dist_matrix, N, filenames):
        ids = np.unique(np.array([self.get_subject_id(f) for f in filenames]),
                        return_inverse=True)[1]
        frr, eer = self.cal_FRR_at_FAR_and_EER(dist_matrix, N, ids)
        rank1 = self.cal_Rank_1(dist_matrix, N, ids)
        return frr, eer, rank1
