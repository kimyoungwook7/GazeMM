import os
import random
import zlib
from types import SimpleNamespace

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from modules.GazeMM.models.resnet import Resnet34Triplet
from modules.utils import utils


class PKImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.data[i]
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i])


class PKBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, P, K, num_batches, rng):
        self.labels = np.asarray(labels)
        self.P = P
        self.K = K
        self.num_batches = num_batches
        self.rng = rng
        self.class_to_idx = {}
        for i, l in enumerate(self.labels):
            self.class_to_idx.setdefault(int(l), []).append(i)
        self.classes = list(self.class_to_idx.keys())

    def _choose_classes(self):
        P = min(self.P, len(self.classes))
        return [int(c) for c in self.rng.choice(self.classes, size=P, replace=False)]

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for c in self._choose_classes():
                idxs = self.class_to_idx[int(c)]
                replace = len(idxs) < self.K
                sel = self.rng.choice(idxs, size=self.K, replace=replace)
                batch.extend(int(s) for s in sel)
            yield batch

    def __len__(self):
        return self.num_batches


class irisRecognition(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on GPU (CUDA)" if self.device.type == 'cuda' else "Running on CPU")


    @staticmethod
    def argument_settings():
        return SimpleNamespace(
            epochs=150,
            patience=10,
            embedding_dimension=512,
            num_human_identities_per_batch=16,
            k_per_identity=4,
            batch_size=32,
            learning_rate=0.075,
            weight_decay=1e-5,
            image_size=224,
        )


    def train_batch_hard(self, dataloader, model, optimizer_model):
        model.train()
        total_loss, n_batches, n_active = 0.0, 0, 0
        for imgs, labels in dataloader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            emb = model(imgs)
            dist = torch.cdist(emb, emb, p=2)

            B = labels.size(0)
            same = labels.unsqueeze(0) == labels.unsqueeze(1)
            eye = torch.eye(B, dtype=torch.bool, device=self.device)
            pos_mask = same & ~eye
            neg_mask = ~same

            d_pos = dist.clone()
            d_pos[~pos_mask] = -1.0
            hardest_pos, _ = d_pos.max(dim=1)

            d_neg = dist.clone()
            d_neg[~neg_mask] = float('inf')
            hardest_neg, _ = d_neg.min(dim=1)

            valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
            if valid.sum() == 0:
                continue

            loss = torch.nn.functional.softplus(hardest_pos[valid] - hardest_neg[valid]).mean()

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            total_loss += loss.item()
            n_batches += 1
            n_active += int(valid.sum().item())

        avg = total_loss / max(1, n_batches)
        print(f"batch-hard loss: {avg:.6f}  (anchors with a valid triplet: {n_active})")
        return avg


    def _build_subspace(self, X, n_components):
        if X.shape[0] < 2:
            return None
        mu = X.mean(axis=0)
        Xc = X - mu
        k = min(n_components, Xc.shape[0] - 1, Xc.shape[1])
        if k <= 0:
            return None
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return mu, Vt[:k]

    @staticmethod
    def _point_to_subspace(x, subspace):
        mu, basis = subspace
        r = x - mu
        proj = basis.T @ (basis @ r)
        return float(np.linalg.norm(r - proj))

    @staticmethod
    def _residuals_to_subspace(X, subspace):
        mu, basis = subspace
        r2 = np.einsum('md,md->m', X, X) - 2.0 * (X @ mu) + float(mu @ mu)
        B = X @ basis.T - (basis @ mu)
        p2 = np.einsum('mk,mk->m', B, B)
        return np.sqrt(np.maximum(r2 - p2, 0.0))

    @staticmethod
    def _min_pointdist(X, Y):
        d = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
        return d.min(axis=1)

    def manifold_distance_matrix(self, emb, ids, n_components=3, min_imgs=3):
        E = emb.detach().cpu().numpy().astype(np.float64)
        N = E.shape[0]
        ids = np.asarray(ids)
        uniq = list(dict.fromkeys(ids.tolist()))
        cls_idx = {c: np.where(ids == c)[0] for c in uniq}

        score_by_class = {}
        for c in tqdm(uniq, desc="Point-to-manifold distances", leave=False):
            members = cls_idx[c]
            m = len(members)
            is_member = (ids == c)
            col = np.empty(N, dtype=np.float64)
            full_sub = self._build_subspace(E[members], n_components)

            nm = np.where(~is_member)[0]
            if len(nm) > 0:
                if full_sub is not None and m >= min_imgs:
                    col[nm] = self._residuals_to_subspace(E[nm], full_sub)
                elif m > 0:
                    col[nm] = self._min_pointdist(E[nm], E[members])
                else:
                    col[nm] = 1e6

            _b = None
            if m >= 3 and (m - 1) >= min_imgs and min(n_components, m - 2, E.shape[1]) >= 1:
                _b = self._loo_residuals_batched(E[members], n_components)
            if _b is not None:
                col[members] = _b
                score_by_class[c] = col
                continue
            for i in members:
                use = members[members != i]
                sub = self._build_subspace(E[use], n_components)
                if sub is not None and len(use) >= min_imgs:
                    col[i] = self._point_to_subspace(E[i], sub)
                elif len(use) > 0:
                    col[i] = float(np.linalg.norm(E[i] - E[use], axis=1).min())
                else:
                    col[i] = 1e6
            score_by_class[c] = col

        M = np.zeros((N, N), dtype=np.float32)
        for j in range(N):
            M[:, j] = score_by_class[ids[j]]
        np.fill_diagonal(M, 0.0)
        return torch.from_numpy(M)

    def _loo_residuals_batched(self, X, n_components):
        m, D = X.shape
        k = int(min(n_components, m - 2, D))
        mu = X.mean(axis=0)
        Xc = X - mu
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        C = Xc @ Vt.T
        dev = torch.device("cpu")
        Ct = torch.from_numpy(np.ascontiguousarray(C)).to(dev)
        r = Ct.shape[1]
        keep = ~torch.eye(m, dtype=torch.bool, device=dev)
        B = Ct.unsqueeze(0).expand(m, m, r)[keep].reshape(m, m - 1, r)
        mean_i = B.mean(dim=1, keepdim=True)
        Bc = B - mean_i
        _, S, Vh = torch.linalg.svd(Bc, full_matrices=False)
        s_top = S[:, 0]
        s_k = S[:, k - 1]
        if bool((s_top <= 0).any()) or bool((s_k < 1e-8 * s_top).any()):
            return None
        W = Vh[:, :k, :]
        ri = Ct - mean_i[:, 0, :]
        p = torch.einsum("mkr,mr->mk", W, ri)
        res2 = (ri * ri).sum(dim=1) - (p * p).sum(dim=1)
        return torch.sqrt(torch.clamp(res2, min=0.0)).cpu().numpy()


    NORM_MODES = ((None, "raw"), ("t", "T-norm"), ("ls", "ls"))
    REPORTED_MODE = "ls"

    def test(self, args, data, data_name, transform, model):
        util = utils()
        model.eval()

        embeddings, buf = [], []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img in tqdm(data, desc="Extracting embeddings"):
                buf.append(transform(img))
                if len(buf) == args.batch_size:
                    x = torch.stack(buf).to(self.device)
                    embeddings.append(model(x).float().cpu())
                    buf = []
            if buf:
                x = torch.stack(buf).to(self.device)
                embeddings.append(model(x).float().cpu())

        emb = torch.cat(embeddings, dim=0)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        N = emb.size(0)

        ids = np.unique(np.asarray([util.get_subject_id(f) for f in data_name]),
                        return_inverse=True)[1]

        _ncomp = getattr(self, "manifold_components", 3)
        _nmin = getattr(self, "manifold_min_imgs", 3)
        dist_matrix = self.manifold_distance_matrix(emb, ids, n_components=_ncomp, min_imgs=_nmin)
        print(f"[match] point-to-manifold matching: subspace dimension {_ncomp}")

        dm = dist_matrix.numpy()
        results = {}
        for mode, label in self.NORM_MODES:
            _dm = dm if mode is None else self._score_norm(dm, ids, mode)
            _f, _e, _r1 = util.evaluate(_dm.copy(), N, data_name)
            results[label] = (_f * 100, _e * 100, _r1 * 100)
            print(f"    {label:<8} FRR@1e-3={_f*100:6.2f}%  EER={_e*100:5.2f}%  Rank-1={_r1*100:6.2f}%")
        return results


    def _score_norm(self, M, ids, mode):
        M = np.array(M, dtype=np.float32, copy=True)
        ids = np.asarray(ids)
        N = M.shape[0]

        s1 = np.empty(N, np.float64)
        s2 = np.empty(N, np.float64)
        _B = max(1, min(N, 4096))
        for a in range(0, N, _B):
            blk = M[a:a + _B].astype(np.float64)
            s1[a:a + _B] = blk.sum(1)
            s2[a:a + _B] = np.einsum('ij,ij->i', blk, blk)
            del blk

        g1 = np.zeros(N, np.float64)
        g2 = np.zeros(N, np.float64)
        cnt = np.empty(N, np.float64)
        for c in np.unique(ids):
            mem = np.where(ids == c)[0]
            sub = M[np.ix_(mem, mem)].astype(np.float64)
            g1[mem] = sub.sum(1)
            g2[mem] = np.einsum('ij,ij->i', sub, sub)
            cnt[mem] = N - len(mem)
            del sub
        cnt = np.maximum(cnt, 1.0)

        mu = (s1 - g1) / cnt
        var = (s2 - g2) / cnt - mu ** 2
        sd = np.sqrt(np.maximum(var, 1e-18))

        if mode == 't':
            sdd = np.where(sd > 1e-9, sd, 1.0).astype(np.float32)
            M -= mu.astype(np.float32)[:, None]
            M /= sdd[:, None]
        elif mode == 'ls':
            s = np.sqrt(np.maximum(mu, 1e-9)).astype(np.float32)
            M /= s[:, None]
            M /= s[None, :]
        else:
            raise ValueError(f"unknown score normalization mode: {mode!r}")

        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 0.0)
        return M


    def _fold_seed(self):
        base = int(getattr(self, "cv_seed", 42))
        return (base + zlib.crc32(self._fold_tag.encode("utf-8"))) % (2 ** 31 - 1)

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    def train_triplet_loss(self, args, train_data_name, train_data_class, train_data,
                           test_data_name, test_data):
        _seed = self._fold_seed()
        self._seed_everything(_seed)

        if getattr(self, "keep_aspect", False) and len(train_data) > 0:
            _w0, _h0 = train_data[0].size
            _tw = max(1, round(args.image_size * _w0 / _h0))
            _sizer = transforms.Resize((args.image_size, _tw))
        else:
            _sizer = transforms.Resize((args.image_size, args.image_size))

        data_transforms = transforms.Compose([
            _sizer,
            transforms.Grayscale(num_output_channels = 3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


        model = Resnet34Triplet(embedding_dimension=args.embedding_dimension,
                                pretrained=True).to(self.device)
        optimizer_model = optim.Adagrad(
            params=model.parameters(),
            lr=args.learning_rate,
            lr_decay=0,
            initial_accumulator_value=0.1,
            eps=1e-10,
            weight_decay=args.weight_decay
        )

        patience = args.patience
        _orig_name, _orig_data, _orig_class = train_data_name, train_data, train_data_class
        _rng = random.Random(42)
        _classes = sorted({int(c) for c in _orig_class}); _rng.shuffle(_classes)
        n_val_cls = max(1, int(round(len(_classes) * 0.15)))
        _val_classes = set(_classes[:n_val_cls])
        _tr_idx  = [i for i, c in enumerate(_orig_class) if int(c) not in _val_classes]
        _val_idx = [i for i, c in enumerate(_orig_class) if int(c) in _val_classes]
        val_data_name  = [_orig_name[i]  for i in _val_idx]
        val_data       = [_orig_data[i]  for i in _val_idx]
        train_data_name  = [_orig_name[i]  for i in _tr_idx]
        train_data       = [_orig_data[i]  for i in _tr_idx]
        train_data_class = np.asarray([_orig_class[i] for i in _tr_idx])
        _n_tr_ids = len(set(int(c) for c in train_data_class))
        print(f"[val split] subject-disjoint 85/15: "
              f"train {len(_tr_idx)} images / {_n_tr_ids} identities, "
              f"validation {len(_val_idx)} images / {n_val_cls} identities")

        labels = [label for _, label in self.NORM_MODES]
        best_val_eer = {lab: float('inf') for lab in labels}
        best_val_frr = {lab: float('inf') for lab in labels}
        es_counter = {lab: 0 for lab in labels}
        best_state = {lab: None for lab in labels}
        best_epoch = {lab: -1 for lab in labels}
        print(f"Early stopping on validation EER (patience {patience}), tracked separately for {', '.join(labels)}")

        P = min(args.num_human_identities_per_batch, _n_tr_ids)
        K = args.k_per_identity
        num_bh_batches = max(1, int(round(5 * len(train_data) / max(1, P * K))))
        bh_dataset = PKImageDataset(train_data, train_data_class, transform=data_transforms)
        bh_loader = torch.utils.data.DataLoader(
            dataset=bh_dataset,
            batch_sampler=PKBatchSampler(train_data_class, P, K, num_bh_batches,
                                         rng=np.random.RandomState(_seed % (2 ** 32))),
        )
        print(f"[batch-hard] P={P} identities x K={K} images = batch size {P*K}; "
              f"{num_bh_batches} batches per epoch (hardest positive and hardest negative, soft margin)")

        print(f"Training for up to {args.epochs} epochs")
        for epoch in range(args.epochs):
            print(f"Epoch {epoch}\n-------------------------------")

            avg_loss = self.train_batch_hard(bh_loader, model, optimizer_model)
            print(f"Epoch {epoch} | mean batch loss: {avg_loss:.6f}")

            val_res = self.test(args, val_data, val_data_name, data_transforms, model)
            cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for lab in labels:
                v_frr, v_eer, _ = val_res[lab]
                if (v_eer < best_val_eer[lab]) or (v_eer == best_val_eer[lab] and v_frr < best_val_frr[lab]):
                    best_val_eer[lab] = v_eer
                    best_val_frr[lab] = v_frr
                    es_counter[lab] = 0
                    best_epoch[lab] = epoch
                    best_state[lab] = cur_state
                else:
                    es_counter[lab] += 1
            print("  [val] " + "  ".join(
                f"{lab}: EER={val_res[lab][1]:.2f}% (best {best_val_eer[lab]:.2f}%, "
                f"{es_counter[lab]}/{patience} epochs without improvement)" for lab in labels))
            if all(es_counter[lab] >= patience for lab in labels):
                print(f"Early stopping at epoch {epoch}: no variant improved for {patience} epochs")
                break
            if all(best_val_eer[lab] < 0.005 and best_val_frr[lab] < 0.005 for lab in labels):
                print(f"[val] validation EER and FRR are 0.00% for every variant (perfect separation); "
                      f"stopping at epoch {epoch}")
                break

        rep = self.REPORTED_MODE
        print(f"\n====== Final evaluation on the test set (best {rep} model) ======")
        if best_state[rep] is not None:
            model.load_state_dict(best_state[rep])
        _f, _e, _r1 = self.test(args, test_data, test_data_name, data_transforms, model)[rep]
        print(f"  best epoch {best_epoch[rep]}:  FRR@1e-3={_f:6.2f}%  EER={_e:5.2f}%  Rank-1={_r1:6.2f}%")
        return _f, _e, _r1


    @staticmethod
    def _pad_to_square(img, fill=127):
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        canvas = Image.new(img.mode, (s, s), fill)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas

    @staticmethod
    def _dataset_tag(filenames):
        common = os.path.commonpath([os.path.abspath(f) for f in filenames])
        return os.path.basename(common.rstrip(os.sep))

    def _preprocess(self, filenames, store_size=256):
        util = utils()

        gate_c = float(getattr(self, "focus_gate_c", 2.0))
        keep_aspect = bool(getattr(self, "keep_aspect", False))

        all_content, all_fn, all_focus = [], [], []
        for fn in tqdm(filenames, total=len(filenames)):
            orig = Image.open(fn).convert('L')
            _w, _h = orig.size
            content = orig.resize((max(1, round(store_size * _w / _h)), store_size))
            focus = float(cv2.Laplacian(np.asarray(content).astype(np.uint8), cv2.CV_64F).var())
            all_content.append(content); all_fn.append(fn); all_focus.append(focus)

        focus_all = np.array(all_focus)

        if len(focus_all):
            lf = np.log(np.maximum(focus_all, 1e-6))
            m = float(np.median(lf)); mad = float(np.median(np.abs(lf - m)))
            thr = (m - gate_c * 1.4826 * mad) if mad > 1e-12 else -np.inf
            keep_flags = lf >= thr
        else:
            keep_flags = np.ones(0, dtype=bool)

        data_name, data_subj, data = [], [], []
        for i in range(len(all_fn)):
            if not bool(keep_flags[i]):
                continue
            c = all_content[i]
            if keep_aspect:
                sq = c
            else:
                sq = self._pad_to_square(c).resize((store_size, store_size))
            data_name.append(all_fn[i]); data_subj.append(util.get_subject_id(all_fn[i])); data.append(sq)

        n_drop = len(filenames) - len(data)
        self.fte_pct = 100.0 * n_drop / max(1, len(filenames))
        print(f"\n[quality] {len(filenames)} images: {len(data)} kept, {n_drop} discarded by the focus gate "
              f"(c={gate_c}); FTE = {self.fte_pct:.2f}%")
        return data, data_name, data_subj


    def run(self, filenames):
        data, data_name, data_class = self._preprocess(filenames)
        if len(data) == 0:
            print("[quality] All images were discarded by the focus gate; increase focus_gate_c to relax it.")
            return None

        args = self.argument_settings()
        k = int(getattr(self, 'cv_folds', 2))
        repeats = int(getattr(self, 'cv_repeats', 1))
        base_seed = int(getattr(self, 'cv_seed', 42))
        n_ids = len(set(data_class))

        all_res = []
        _name_of = [os.path.abspath(str(n)) for n in data_name]
        for _tag, _tr_fns, _te_fns in (getattr(self, "fixed_splits", None) or []):
            _te_set = {os.path.abspath(f) for f in _te_fns}
            te_idx = [i for i, n in enumerate(_name_of) if n in _te_set]
            if len(te_idx) < 2 or len(te_idx) >= len(_name_of) - 1:
                print(f"[CV] split {_tag}: only {len(te_idx)} test image(s); skipping")
                continue
            seed_key = f"[CV] 공용분할 {_tag} (Subject-disjoint, open-set)"
            _rep_part, _fold_part = _tag.split("·", 1)
            label = f"repeat {_rep_part}, {_fold_part}"
            tag = f"[CV] {label} (subject-disjoint, open-set)"
            tr_idx = [i for i in range(len(data_class)) if i not in set(te_idx)]
            tr_name = [data_name[i] for i in tr_idx]; tr_data = [data[i] for i in tr_idx]
            te_name = [data_name[i] for i in te_idx]; te_data = [data[i] for i in te_idx]
            tr_cls = LabelEncoder().fit_transform(np.array([data_class[i] for i in tr_idx]))
            n_tr_ids = len({data_class[i] for i in tr_idx}); n_te_ids = len({data_class[i] for i in te_idx})
            print("\n" + "#" * 70)
            print(f"# {tag}")
            print(f"#   train: {len(tr_idx)} images / {n_tr_ids} identities   "
                  f"test: {len(te_idx)} images / {n_te_ids} identities   "
                  f"({n_ids} identities in total, subject-disjoint)")
            print("#" * 70)
            self._fold_tag = f"{self._dataset_tag(data_name)}|{seed_key}"
            r = self.train_triplet_loss(args, tr_name, tr_cls, tr_data, te_name, te_data)
            all_res.append(r)
            print(f"[{label}]  FRR@1e-3={r[0]:.2f}%  EER={r[1]:.2f}%  Rank-1={r[2]:.2f}%")

        _ok = [r for r in all_res if not any(np.isnan(float(v)) for v in r)]
        if not _ok:
            print("[result] No valid fold; no result for this dataset.")
            return None
        if len(_ok) < len(all_res):
            print(f"[result] Excluded {len(all_res) - len(_ok)} fold(s) with undefined metrics; "
                  f"averaging over the remaining {len(_ok)}.")
        fr, ee, rr = ([x[0] for x in _ok], [x[1] for x in _ok], [x[2] for x in _ok])
        res = (float(np.mean(fr)), float(np.std(fr)), float(np.mean(ee)), float(np.std(ee)),
               float(np.mean(rr)), float(np.std(rr)))
        print("\n" + "=" * 78)
        print(f"[result] {repeats}x{k}-fold cross-validation, {len(_ok)} evaluations (seed {base_seed})")
        print(f"    FRR@1e-3={res[0]:6.2f}%±{res[1]:4.2f}  EER={res[2]:5.2f}%±{res[3]:4.2f}  "
              f"Rank-1={res[4]:6.2f}%±{res[5]:4.2f}")
        print("=" * 78)
        return res

