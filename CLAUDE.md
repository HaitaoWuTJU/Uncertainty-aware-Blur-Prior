# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Official implementation of "Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior" (CVPR 2025). Trains a brain (EEG/MEG) encoder to align with frozen CLIP image embeddings via a CLIP-style contrastive loss, evaluated as a retrieval task (top-1/top-5 accuracy, mAP). The UBP contribution: per-sample uncertainty estimated from the contrastive logits selects how strongly the *target image* is fovea-blurred.

## Scope: EEG only

This fork targets **THINGS-EEG only**. `base/data_meg.py`, `preprocess/process_meg.py`, `configs/meg/` and the MEG branches in `main.py` are kept as legacy in case they are needed later — do not maintain, update, or test them, and do not extend changes into them unless explicitly asked. When editing shared code, keep the MEG paths compiling but leave their behaviour alone.

## Division of labor — do not launch trainings

Claude edits code/configs and **writes** batch-job scripts; the user **submits** them and plans when runs happen (a full run is hours-long). So:

- Never run `main.py` as a real training job, and never `bsub` anything. Write the job script, say it's ready and how to submit it, and stop.
- Cheap, seconds-long checks on the login node are fine and expected: `python -c "import main"`, config resolution, shape math, `--epoch 1` only if the user asks for it explicitly.
- When a change needs verification that only a real run can give, say so plainly instead of starting one.

This is DTU HPC with the **LSF** scheduler (`bsub`, `bstat`/`bjobs`, `bkill`; GPU queues include `gpua100`, `gpuv100`, `gpul40s`, `gpuh100`). Put job scripts in `scripts/` as `#BSUB`-header shell scripts that `cd` to the repo root, activate the `ubp` env, and run `main.py`; the user submits with `bsub < scripts/<job>.sh`. Prefer one script per logical experiment (or an LSF job array over subjects/seeds) over a script that loops through the whole sweep serially, so the user can queue and cancel pieces independently.

## Environment

Always run Python inside the `ubp` conda env, with CUDA (never force CPU):

```bash
conda run -n ubp python main.py ...   # or `conda activate ubp` first
```

The installed env has **drifted from `requirements.txt`** — it is Python 3.10.20 with numpy 2.2.6, not the documented Python 3.8 / numpy 1.24.4. Treat `requirements.txt` as historical, `conda list -n ubp` as truth.

Current versions: Python 3.10.20, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, mne 1.12.1, torch 2.4.1+cu121.

Resolved 2026-08-13: `pandas 2.0.3` was built against the numpy 1.x ABI and raised `ValueError: numpy.dtype size changed ... Expected 96 from C header, got 88` on plain `import pandas`, which killed preprocessing (via `sklearn`) and training (via `base/inpating_data.py`). Upgrading to `pandas>=2.2.2` fixed it. Note the failure mode: `sklearn/utils/fixes.py` wraps its pandas import in `try/except ImportError`, and an ABI mismatch raises `ValueError`, so the guard does not catch it — an unrelated-looking sklearn traceback can mean a numpy/pandas ABI skew.

mne 1.12 still supports every call `process_eeg_whiten.py` makes; `raw.pick_channels(..., ordered=True)` is legacy but functional.

## Commands

Training — reference for composing batch scripts, not for Claude to execute (see `scripts/exp.sh` for the full sweep; `--dataset` must match the config dir):

```bash
python main.py --config configs/eeg/ubp.yaml --dataset eeg --subjects sub-01 --seed 0 \
  --exp_setting intra-subject --brain_backbone EEGProjectLayer --vision_backbone RN50 \
  --epoch 50 --lr 1e-4
```

- `--exp_setting intra-subject` uses lr 1e-4; `inter-subject` (leave-one-subject-out) uses lr 1e-5.
- `--brain_backbone`: any class in `base/eeg_backbone.py` (`EEGProjectLayer`, `Shallownet`, `Deepnet`, `EEGnet`, `TSconv`).
- `--vision_backbone`: a key of `pretrain_map` in `main.py` (`RN50`, `RN101`, `ViT-B-16` … `ViT-bigG-14`); it also fixes `z_dim`.
- EEG has 10 subjects (`sub-01`…`sub-10`), MEG has 4 (`sub-01`…`sub-04`).

Preprocessing — submit as batch jobs, resize first (it produces the image tree the EEG script reads):

```bash
bsub < scripts/lsf_resize_images.sh                            # ~17k images, 500x500 -> 224x224
bsub -w "done(ubp_resize)" < scripts/lsf_preprocess_eeg.sh     # array [1-10], one subject per task
```

The underlying commands (CWD must be the repo root):

```bash
python preprocess/process_resize.py --type eeg        # idempotent; --overwrite to redo
python preprocess/process_eeg_whiten.py --subject 1   # ~25 GB peak RSS, CPU only
```

## Outputs

`save_dir` (configs) is `/work3/s193209/data/ubp_exp`. `main.py` builds a `TensorBoardLogger(save_dir, name=config['name'], version="<subjects>_seed<seed>")`, so everything for one run lands in:

```
/work3/s193209/data/ubp_exp/eeg_intra-subject_ubp_EEGProjectLayer_ViT-H-14/sub-01_seed0/
├── events.out.tfevents.*   # all self.log() metrics
├── ubp.yaml                # copy of the config actually used (shutil.copy of --config)
├── checkpoints/            # last.ckpt + one epoch=..-step=...ckpt (~65 MB each)
└── test_results.json       # trainer.test() return value
```

`name` is the interpolation `${dataset}_${exp_setting}_{baseline|ubp}_${brain_backbone}_${vision_backbone}`, so backbone/setting changes get their own tree, and `version` isolates subject+seed. Logged metrics: `train_loss`, `train_top{1,5}_acc`, `low`/`medium`/`high` (the UBP match-label counts per epoch), `val_*`, and at test time `test_top{1,5}_acc`, `mAP`, `similarity`.

Do **not** write outputs into `/work3/s193209/data/thesis_specific` — that tree is unrelated data and off-limits.

Gotchas:
- `trainer.fit(..., ckpt_path='last')` means re-running the same name/subject/seed **resumes**; delete the version dir for a clean run.
- `intra-subject` tests `ckpt_path='last'`; `inter-subject` tests `ckpt_path='best'`, but `ModelCheckpoint(save_last=True)` is constructed with **no `monitor`**, so "best" does not track `val_top1_acc` — inter-subject test numbers are not from the peak-validation epoch.
- The CLIP feature cache (`<feature_dir>/<BlurClass>/<name>_<mode>.pt`, ~275 MB for ViT-H-14) is keyed by `name`, which includes `brain_backbone` even though image features do not depend on it — so switching brain backbone re-encodes needlessly, while changing blur hyperparameters does *not* invalidate it (see below). Writes go through a temp file + `os.replace`, so concurrent array tasks cannot tear it, but they will each redo the encoding — warm the cache with a single run first.

## EEG config variants

| config | channels | `c_num` | trial averaging | train samples/subject |
|---|---|---|---|---|
| `ubp.yaml` | 17 occipital/parietal | 17 | on | 16540 |
| `ubp_allch.yaml` | all 63 | 63 | on | 16540 |
| `ubp_allch_noavg.yaml` | all 63 | 63 | **off** | 66160 |
| `baseline.yaml` | 17 | 17 | on | 16540 (no blur, `uncertainty_aware: False`) |

`selected_ch: False` keeps all 63 channels (the `.pt` files store them in `chan_order`, so no reindexing is needed); `c_num` must match or `EEGProjectLayer`'s `input_dim` is wrong. Each variant has its own `name`, so output dirs do not collide — important because `ckpt_path='last'` would otherwise make one variant resume from another's checkpoint.

**`train_avg: False` trains on individual trials.** `load_data` reshapes `(16540, 4, 63, 250) -> (66160, 63, 250)`, so every repetition is its own sample (65 steps/epoch instead of 17). `batch['eeg_mean']` still carries the per-image average, but `forward` does not use it. Consequence for the contrastive loss: a 1024-sample batch drawn from 66160 trials of 16540 images contains ~32 duplicate-image pairs, which appear as false negatives in `ClipLoss`.

**`test_avg: False` invalidates the batch-diagonal metrics.** The test loader is `shuffle=False` over image-major data, so a batch of 200 holds only ~3 unique images, while `test_step` assumes `label = arange(batch_size)` and reads the similarity diagonal. `test_top1_acc`, `test_top5_acc`, `mAP` and `similarity` are therefore meaningless whenever `test_avg: False`. Use the `*_gallery` metrics instead (below), which rank every trial against all 200 unique test images and reduce to the same values when `test_avg: True`.

`data.feature_name` decouples the CLIP cache key from the run `name`, so channel/averaging/brain-backbone variants share one encode. It resolves to `ViT-H-14_ua_k51_g3_c6` for the UBP configs (vision backbone + blur hyperparameters, which is what image features actually depend on) and `ViT-H-14` for baseline. Unlike the old `name`-based key, changing `blur_kernel_size`/`system_g`/`c` now correctly produces a new cache file.

## Embedding export

With `save_embeddings: true` (all EEG configs), `on_test_epoch_end` writes `test_embeddings.pt` next to the event file:

```python
{'eeg_z': (n_reps, n_images, z_dim),   # (1, 200, 1024) averaged; (80, 200, 1024) no-avg
 'img_z': (n_images, z_dim),           # (200, 1024) CLIP gallery, one row per image
 'img_paths': [...],                   # len n_images, aligned with dim 1 / dim 0
 'n_reps': int, 'n_images': int, 'normalized': True, 'dim_order': str}
```

`eeg_z` is L2-normalized (as used for retrieval). `n_reps` is inferred from how many times the first image path repeats, relying on repetitions being contiguous in dataset order. The same pass logs `test_top1_acc_gallery`, `test_top5_acc_gallery` and `mAP_gallery`.

## Watching runs

`/bin/bash scripts/tensorboard.sh [port]` on a login node, then tunnel from the laptop to **that same** login node (`ssh -L 6006:localhost:6006 s193209@login.hpc.dtu.dk`; `login.hpc.dtu.dk` round-robins, so check `hostname`).

The `ubp` env's own tensorboard **2.14 cannot run as a server**: it needs `pkg_resources` (gone in setuptools ≥ 81) and `np.string_` (gone in numpy 2). The *writer* path used by `TensorBoardLogger` is unaffected — `torch.utils.tensorboard` only imports the top-level package — so training logs fine. The script therefore runs a standalone tensorboard from `/work3/s193209/pyenvs/tb` via `PYTHONPATH`, pinned to **2.19.0** (2.21's protobuf gencode requires protobuf 6.x; the env has 5.28.3). Do not upgrade tensorboard inside `ubp` while a job is running — it would swap files under the live process.

## Data layout (this machine)

EEG data lives on `/work3`, not in the repo's `data/` dir. Paths come from env vars with these defaults:

| Env var | Default | Contents |
|---|---|---|
| `UBP_RAW_EEG_DIR` | `/work3/s193209/data/eeg_raw` | `sub-01..10/ses-01..04/raw_eeg_{training,test}.npy` (138 GB, verified complete) |
| `UBP_IMAGE_SRC_DIR` | `/work3/s193209/data/images` | shared THINGS images, `training_images/` + `test_images/` at 500×500 |
| `UBP_IMAGE_RESIZE_DIR` | `/work3/s193209/data/images/Image_set_Resize` | 224×224 copies, `{train,test}_images/` (done: 16540 + 200) |
| `UBP_DATA_ROOT` | `/work3/s193209/data/eeg_preprocessed_250hz/ubp_format` | EEG-derived artefacts only |

**Images live in the shared `images/` dir; `ubp_format/` is for EEG only.** They are deliberately not colocated:

```
/work3/s193209/data/images/Image_set_Resize/{train_images,test_images}/   # shared across formats
/work3/s193209/data/eeg_preprocessed_250hz/ubp_format/
├── Preprocessed_data_250Hz_whiten/sub-XX/{train,test}.pt
└── Image_feature/<BlurClass>/     # CLIP feature cache, created on first training run
```

That split means UBP's upstream `<data_dir>/../Image_set_Resize` convention no longer holds, so `base/data_eeg.py` and `base/data_meg.py` read `data.image_dir` and `data.feature_dir` from the config, falling back to the old `<data_dir>/../` layout when those keys are absent (the MEG configs still rely on the fallback). `configs/eeg/*.yaml` set all three paths absolutely.

The `.pt` files store image paths **relative to `image_dir`** (via `os.path.relpath`), so preprocessing and training must point at the same dir — moving the resized images means regenerating the `.pt` files or updating `image_dir` to match.

`ubp_format` sits beside the existing `thesis_format`/`thought2text_format` dirs, which other pipelines use — don't write into those.

`process_eeg_whiten.py` derives its array shapes from the loaded data instead of the upstream hardcoded THINGS-EEG2 constants (`(33080, 2)`, `white_data.shape[1]*2`, `(16540, 4)`, `(200, 80)`), and raises if training condition ids are not a dense `1..N` range or if images appear in a varying number of sessions. For the current data these derive to exactly the original values: 200×80 test, 16540 conditions × 4 repetitions train. Do not re-hardcode them — the dense-id check guards a silent corruption where `np.where` returns nothing and the previous image's data is reused.

Two naming mismatches the scripts absorb, worth knowing if you touch them:
- Raw train split is `raw_eeg_training.npy`, but `mode` is `'train'` internally; `epoch_data` tries `training` then `train`.
- Source image dir is `training_images`, but `process_eeg_whiten.py` and the stored `.pt` paths expect `train_images`; `process_resize.py` renames it via `SPLIT_DIRS`. The source tree also holds unrelated PNGs (`image_text_plots/`), so the resize walk is restricted to the two split dirs rather than the whole root.

There is no test suite and no linter. Per `.clinerules`, after editing a script do a smoke check (e.g. `conda run -n ubp python -c "import main"`) rather than a full run.

## Architecture

**Config-driven instantiation.** `configs/{eeg,meg}/{baseline,ubp}.yaml` are OmegaConf files with `${...}` interpolations (`${brain_backbone}`, `${z_dim}`, `${epoch}`, `${lr}`, `${dataset}`, `${exp_setting}`). `base.utils.update_config` copies **every** argparse attribute onto the top-level config, which is what resolves those interpolations — so adding a new `${var}` to a yaml requires a matching `--var` in `main.py`'s parser. Models are built by `instantiate_from_config` from `target` + `params` strings (`base.eeg_backbone.${brain_backbone}`, `base.inpating_data.FoveaBlur`), so config strings are import paths — renaming a class breaks the yaml.

`baseline.yaml` vs `ubp.yaml` differ only in `uncertainty_aware` (False/True) and `blur_type` (`DirectT` identity vs `FoveaBlur`).

**The UBP loop** (this is the part that spans files):

1. `PLModel.forward` (`main.py`) computes `ClipLoss` and keeps the diagonal of `logits_per_image` — each sample's similarity to its own image.
2. It EMA-smooths that into `self.sim[idx]` (`gamma=0.3`) and assigns `match_label[idx]` ∈ {0,1,2} by whether the similarity falls outside a two-sided normal CI (`alpha=0.05`) of the batch: above upper bound → `0` = "low" (easy/certain), below lower → `2` = "high", else `1` = "medium".
3. At the last training batch of the epoch it writes the array back: `self.trainer.train_dataloader.dataset.match_label = self.match_label`.
4. `EEGDataset.__getitem__` / `MEGDataset.__getitem__` (`base/data_eeg.py`, `base/data_meg.py`) reads `match_label[index]` and returns the pre-encoded CLIP feature for the corresponding blur strength (`low`/`medium`/`high`). Val/test always use `medium`.

So difficulty is fed back as *target blur*, not as a loss weight. Blur strengths are the config `blur_kernel_size` shifted by `±--c` (default 6); `FoveaBlur` (`base/inpating_data.py`) alpha-blends a Gaussian blur with the sharp image using a radial mask whose falloff curve is `curve_type` (`exp` with `system_g`, plus `linear`/`quadratic`/`log`/`brachistochrone`).

**Image-feature cache — a common footgun.** Datasets encode all (blurred) images with frozen CLIP once and cache to
`<data_dir>/../Image_feature/<BlurClassName>/<config name>_<mode>.pt`.
The filename is derived only from the blur *class* name and `name` (dataset/exp_setting/baseline-or-ubp/backbones) — it does **not** include `blur_kernel_size`, `system_g`, `curve_type`, or `c`. Changing any blur hyperparameter silently reuses stale features; delete the cache file to force re-encoding. First run of a new combination loads CLIP onto a GPU and encodes 3× the images (low/medium/high).

**Data loading.** `load_eeg_data`/`load_meg_data` return `(train, val, test)`. For `intra-subject` there is no separate val set — the test loader is returned twice, and early stopping monitors `train_loss`. For `inter-subject`, train/val come from the *other* subjects (this mutates `config['data']['subjects']` in place) and early stopping monitors `val_top1_acc`. EEG configs select 17 occipital/parietal channels via `data.selected_ch` (must match `c_num: 17`); MEG uses all 271 channels (`selected_ch: False`, `c_num: 271`). `train_avg`/`test_avg` average repeated trials per image. `timesteps` slices the epoch window and, with `c_num`, determines `EEGProjectLayer`'s input dim.

**Device selection.** `base.utils.get_device('auto')` shells out to `nvidia-smi` and picks a GPU by free memory then temperature. `Trainer` runs single-device (`devices=[device]`) despite `DDPStrategy`. Two environment overrides were added because that logic is wrong under a scheduler: `nvidia-smi` reports *physical* indices while LSF's `CUDA_VISIBLE_DEVICES` makes torch renumber from 0, so `'auto'` could return an index torch cannot see. `get_device` now returns `0` when `CUDA_VISIBLE_DEVICES` is set, and `UBP_GPU` overrides the choice outright (the job scripts set `UBP_GPU=0`). Note `nvidia-smi` does not exist on the login node, so bare `'auto'` cannot run there at all.

**Model weights cache.** `HF_HOME=/work3/s193209/huggingface_cache` — big weights must never land in `$HOME` (no quota). `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` is already cached there and loads with `HF_HUB_OFFLINE=1`; the job scripts export `HF_HOME`/`HF_HUB_CACHE`/`TORCH_HOME` explicitly rather than relying on inherited env.

**`update_config` clobbers config keys with `None`.** Its second loop does `config[key] = getattr(args, key)` for *every* argparse attribute unconditionally. So adding a new CLI flag whose default is `None` will silently overwrite a same-named yaml key with `None` — give new flags a real default, or set the value only in yaml.

**Outputs & resume.** TensorBoard logs, the copied config, checkpoints and `test_results.json` go to `exp/<name>/<subjects>_seed<seed>/`. `main.py` calls `trainer.fit(..., ckpt_path='last')`, so re-running the same name/subject/seed **resumes** from `last.ckpt` instead of starting over — remove the version dir for a clean run. `exp/` and `data/` are gitignored.

## Conventions

From `.clinerules`:
- Work on `dev`; never commit to `main` or push to `upstream`.
- Keep edits scoped to the relevant files; don't refactor unrelated code unasked.
- Be concise; execute directly, but loop the user in on larger changes.
