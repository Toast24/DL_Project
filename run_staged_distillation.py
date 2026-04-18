"""
Run staged student compression:
1) Train/evaluate 53M distilled student.
2) Compare against run21 baseline with max-drop gate.
3) If gate passes, train/evaluate 43M re-distilled student.
4) Compare 43M against run21 with the same gate.

Usage:
  python run_staged_distillation.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def safe_exists(path):
    return Path(path).exists()


def main():
    py = sys.executable

    baseline_dir = "outputs/run21_resplit_15es"
    run53_dir = "outputs/run31_distill_53m_15ep"
    run43_dir = "outputs/run32_redistill_43m_15ep"

    cfg53 = "configs/ssvp_student_53m_distill.yaml"
    cfg43 = "configs/ssvp_student_43m_redistill.yaml"

    data_root = "data/cable_resplit/cable"
    max_drop = "7.5"

    summary = {
        "baseline_dir": baseline_dir,
        "run53": {"dir": run53_dir, "completed": False, "gate_pass": False},
        "run43": {"dir": run43_dir, "completed": False, "gate_pass": False, "skipped": True},
    }

    # Stage 1: 53M distillation run
    run_cmd([
        py,
        "run_full_pipeline.py",
        "--config", cfg53,
        "--data_root", data_root,
        "--output_dir", run53_dir,
        "--epochs", "15",
        "--num_vis_samples", "5",
    ])
    summary["run53"]["completed"] = True

    run53_report = f"{run53_dir}/comparison_to_run21.json"
    try:
        run_cmd([
            py,
            "compare_against_baseline.py",
            "--baseline_dir", baseline_dir,
            "--candidate_dir", run53_dir,
            "--max_drop", max_drop,
            "--report_path", run53_report,
        ])
        summary["run53"]["gate_pass"] = True
    except subprocess.CalledProcessError:
        summary["run53"]["gate_pass"] = False

    # Stage 2: 43M re-distillation (conditional)
    if summary["run53"]["gate_pass"]:
        summary["run43"]["skipped"] = False

        # Ensure 53M teacher checkpoint exists for stage-2 distillation.
        teacher_ckpt_53 = f"{run53_dir}/best_model.pth"
        if not safe_exists(teacher_ckpt_53):
            raise FileNotFoundError(
                f"Expected 53M teacher checkpoint for re-distillation not found: {teacher_ckpt_53}"
            )

        run_cmd([
            py,
            "run_full_pipeline.py",
            "--config", cfg43,
            "--data_root", data_root,
            "--output_dir", run43_dir,
            "--epochs", "15",
            "--num_vis_samples", "5",
        ])
        summary["run43"]["completed"] = True

        run43_report = f"{run43_dir}/comparison_to_run21.json"
        try:
            run_cmd([
                py,
                "compare_against_baseline.py",
                "--baseline_dir", baseline_dir,
                "--candidate_dir", run43_dir,
                "--max_drop", max_drop,
                "--report_path", run43_report,
            ])
            summary["run43"]["gate_pass"] = True
        except subprocess.CalledProcessError:
            summary["run43"]["gate_pass"] = False

    out_path = "outputs/staged_distillation_summary.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nStaged distillation complete.")
    print(f"Summary: {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
