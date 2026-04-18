# Demo Tutorial: Run21 + LLM Compression From Scratch

This tutorial shows exactly how to run the main demo on a fresh machine:

- Train (or reuse) the run21-quality model.
- Run noisy-folder segmentation + captioning.
- Export visualizations, masks, and prompts/captions.

## 1) Hardware Requirements

### Minimum (works, slower)

- OS: Windows 10/11, Linux, or macOS.
- Python: 3.10+.
- RAM: 16 GB.
- GPU: CUDA-capable NVIDIA GPU with 8 GB VRAM.
- Disk: at least 25 GB free for data, checkpoints, and outputs.

### Recommended (for smoother training)

- RAM: 32 GB.
- GPU: 12 GB+ VRAM.
- Disk: 50 GB+.

Notes:

- Training in this repo is GPU-only.
- Demo inference can run on CPU but will be slower.

## 2) Clone and Environment Setup (PowerShell)

```powershell
git clone [<your_repo_url>](https://github.com/Toast24/DL_Project/)
cd ssvp

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Download MVTec AD (Official)

Download from:

- https://www.mvtec.com/company/research/datasets/mvtec-ad

After extraction, ensure cable category folders exist under project root as:

- cable/train/good/
- cable/test/good/
- cable/test/<defect_type>/
- cable/ground_truth/<defect_type>/

## 4) Build the Resplit Dataset (70/20/10)

```powershell
python prepare_cable_split.py --source cable --output_root data/cable_resplit --seed 42
```

Expected output:

- data/cable_resplit/cable/
- data/cable_resplit/cable/split_summary.txt

## 5) Produce Run21 Baseline Checkpoint

### Option A: Reproduce from scratch

```powershell
python run_full_pipeline.py --config configs/default.yaml --data_root data/cable_resplit/cable --output_dir outputs/run21_resplit_15es --epochs 15 --num_vis_samples 5
```

This produces:

- outputs/run21_resplit_15es/best_model.pth
- outputs/run21_resplit_15es/eval_results/results.json
- outputs/run21_resplit_15es/noise_eval/robustness_report.json
- outputs/run21_resplit_15es/visualizations_with_captions/

### Option B: Reuse existing run21 checkpoint

If outputs/run21_resplit_15es/best_model.pth already exists, you can skip training.

## 6) Run the Main Demo (Run21 + LLM Compression)

INT8 text compression for captioning is enabled by default in this demo script.

### Demo on a sample noisy folder

```powershell
python live_demo_noisy_folder.py --input_folder cable/test/combined --output_dir outputs/live_demo --checkpoint outputs/run21_resplit_15es/best_model.pth --category cable --recursive
```

### Demo on your own noisy folder

```powershell
python live_demo_noisy_folder.py --input_folder <PATH_TO_NOISY_IMAGES> --output_dir outputs/live_demo_custom --checkpoint outputs/run21_resplit_15es/best_model.pth --category cable --recursive --max_images 0
```

Optional quality mode (slower):

```powershell
python live_demo_noisy_folder.py --input_folder <PATH_TO_NOISY_IMAGES> --output_dir outputs/live_demo_custom_prompt --checkpoint outputs/run21_resplit_15es/best_model.pth --category cable --recursive --caption_finetune
```

## 7) Understand Demo Outputs

Each processed image gets multiple artifacts under:

- outputs/live_demo*/visualizations_with_captions/

Per-image files:

- *.viz.png: visualization panel with score and caption.
- *.heatmap.png: normalized anomaly heatmap.
- *.mask.png: binary mask from pixel threshold.
- *.overlay.png: red mask overlay on image.
- *.txt: generated caption/prompt text.

Run-level summary:

- outputs/live_demo*/demo_summary.json

## 8) Quick Verification Commands

```powershell
Get-ChildItem outputs/live_demo -Recurse -File | Select-Object -ExpandProperty FullName
Get-Content outputs/live_demo/demo_summary.json
```

## 9) Common Issues

- Missing checkpoint:
  - Ensure outputs/run21_resplit_15es/best_model.pth exists or pass --checkpoint explicitly.
- No images found:
  - Check --input_folder path and extensions.
- Slow captioning:
  - Keep INT8 compression enabled (default) and reduce --max_images.
- Training fails on CPU-only machine:
  - Use a CUDA-capable GPU system for training.

## 10) What To Read Next

1. [RUNALLEXPS.md](RUNALLEXPS.md) for every experiment command.
2. [RESULTS.md](RESULTS.md) for the full outcomes and analysis.
3. [README.md](README.md) for project overview and script map.
