"""Run all three inverse-physics methods and produce a comparison report.

Methods:
  1. Baseline — derivative-free golden-section search
  2. NN+FD   — NN with finite-difference physics gradient
  3. NN+AD   — NN with full Taichi autodiff through simulation

Usage:
  python code/eval_compare.py [--epochs 30] [--baseline_iters 16] [--tiny]
"""
import argparse
import os
import subprocess
import sys
import time
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_ROOT, "Inversive", "code")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_DIR_AD = os.path.join(PROJECT_ROOT, "data_ad")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20,
                    help="training epochs for NN methods")
parser.add_argument("--baseline_iters", type=int, default=16,
                    help="golden-section search iterations")
parser.add_argument("--tiny", action="store_true",
                    help="use tiny simulation config")
parser.add_argument("--skip_gen_target", action="store_true",
                    help="skip target data generation")
args = parser.parse_args()


def run_step(step_name, cmd):
    """Run a subprocess command; return elapsed seconds and success flag."""
    print("\n" + "=" * 72)
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print("=" * 72)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=CODE_DIR,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    elapsed = time.perf_counter() - t0
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    success = result.returncode == 0
    status = "OK" if success else f"FAIL (rc={result.returncode})"
    print(f"\n  [{status}] {step_name} finished in {elapsed:.1f}s")
    return elapsed, success, result.stdout


def ensure_target_data():
    """Generate target trajectory if it doesn't exist."""
    target_path = os.path.join(DATA_DIR, "target_trajectory.npz")
    if os.path.exists(target_path) and not args.skip_gen_target:
        print(f"Target data exists: {target_path}")
        # Read and report E_true
        data = np.load(target_path)
        print(f"  E_true = {float(data['E_true']):.1f}, "
              f"nu_true = {float(data['nu_true']):.3f}, "
              f"n_steps = {data['h'].shape[0]}")
        return True

    print("Generating target trajectory ...")
    cmd = [sys.executable, "gen_target_data.py"]
    if args.tiny:
        cmd += ["--warmup_steps", "20"]
    elapsed, ok, out = run_step("Generate target data", cmd)
    return ok


def run_baseline():
    """Run derivative-free baseline search."""
    print("\n" + "-" * 40)
    print("  Running BASELINE (derivative-free search)")
    print("-" * 40)
    cmd = [
        sys.executable, "optimize_E_search.py",
        "--iters", str(args.baseline_iters),
    ]
    elapsed, ok, out = run_step("Baseline search", cmd)

    result_path = os.path.join(DATA_DIR, "E_search_result.npz")
    if os.path.exists(result_path):
        data = np.load(result_path)
        best_E = float(data["best_E"])
        best_loss = float(data["best_loss"])
        E_true = float(data["E_true"])
        print(f"  Baseline result: E_pred={best_E:.4f}, "
              f"|E-E*|={abs(best_E - E_true):.4f}, "
              f"loss={best_loss:.6e}")
        return {
            "method": "baseline",
            "E_pred": best_E,
            "E_true": E_true,
            "E_error": abs(best_E - E_true),
            "loss": best_loss,
            "time": elapsed,
            "success": ok,
        }
    return {"method": "baseline", "success": False, "error": "result file not found"}


def run_nn_fd():
    """Run NN+FD training and inference."""
    print("\n" + "-" * 40)
    print("  Running NN+FD (finite-difference gradient)")
    print("-" * 40)

    train_cmd = [
        sys.executable, "inverse_train.py",
        "--train",
        "--epochs", str(args.epochs),
    ]
    if args.tiny:
        train_cmd.append("--tiny")

    elapsed_train, ok_train, _ = run_step("NN+FD training", train_cmd)
    if not ok_train:
        return {"method": "nn_fd", "success": False, "error": "training failed"}

    infer_cmd = [sys.executable, "inverse_train.py", "--infer"]
    elapsed_infer, ok_infer, _ = run_step("NN+FD inference", infer_cmd)

    train_log_path = os.path.join(DATA_DIR, "training_log.npz")
    pred_path = os.path.join(DATA_DIR, "predicted_trajectory.npz")

    result = {
        "method": "nn_fd",
        "success": ok_train and ok_infer,
        "train_time": elapsed_train,
        "infer_time": elapsed_infer,
    }

    if os.path.exists(train_log_path):
        log = np.load(train_log_path)
        e_pred = float(log["E_pred"][-1])
        e_true = float(log["E_true"])
        final_loss = float(log["loss"][-1])
        result["E_pred"] = e_pred
        result["E_true"] = e_true
        result["E_error"] = abs(e_pred - e_true)
        result["loss"] = final_loss
        result["n_epochs"] = int(log["epoch"][-1]) + 1

    if os.path.exists(pred_path):
        pred = np.load(pred_path)
        result["mse_h"] = float(pred["mse_h"])
        result["mse_s"] = float(pred["mse_s"])
        result["mse_F"] = float(pred["mse_F"])

    return result


def run_nn_ad():
    """Run NN+AD training and inference."""
    print("\n" + "-" * 40)
    print("  Running NN+AD (full autodiff through simulation)")
    print("-" * 40)

    train_cmd = [
        sys.executable, "inverse_train_ad.py",
        "--train",
        "--epochs", str(args.epochs),
    ]
    if args.tiny:
        train_cmd.append("--tiny")

    elapsed_train, ok_train, stdout_train = run_step("NN+AD training", train_cmd)
    if not ok_train:
        return {"method": "nn_ad", "success": False,
                "error": "training failed", "stdout": stdout_train[-2000:]}

    infer_cmd = [sys.executable, "inverse_train_ad.py", "--infer"]
    elapsed_infer, ok_infer, _ = run_step("NN+AD inference", infer_cmd)

    train_log_path = os.path.join(DATA_DIR_AD, "training_log.npz")
    pred_path = os.path.join(DATA_DIR_AD, "predicted_trajectory.npz")

    result = {
        "method": "nn_ad",
        "success": ok_train and ok_infer,
        "train_time": elapsed_train,
        "infer_time": elapsed_infer,
    }

    if os.path.exists(train_log_path):
        log = np.load(train_log_path)
        e_pred = float(log["E_pred"][-1])
        e_true = float(log["E_true"])
        final_loss = float(log["loss"][-1])
        result["E_pred"] = e_pred
        result["E_true"] = e_true
        result["E_error"] = abs(e_pred - e_true)
        result["loss"] = final_loss
        result["n_epochs"] = int(log["epoch"][-1]) + 1

    if os.path.exists(pred_path):
        pred = np.load(pred_path)
        result["mse_h"] = float(pred["mse_h"])
        result["mse_s"] = float(pred["mse_s"])
        result["mse_F"] = float(pred["mse_F"])

    return result


def save_results(results):
    """Persist comparison results to JSON and NPZ."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    # JSON summary
    json_path = os.path.join(REPORT_DIR, "comparison_summary.json")
    # Convert numpy values for JSON
    json_safe = []
    for r in results:
        entry = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                entry[k] = float(v)
            else:
                entry[k] = v
        json_safe.append(entry)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, indent=2, ensure_ascii=False)
    print(f"\nComparison summary saved to {json_path}")

    # NPZ for easy loading
    npz_path = os.path.join(REPORT_DIR, "comparison_results.npz")
    np.savez(npz_path,
             methods=np.array([r["method"] for r in results]),
             E_error=np.array([r.get("E_error", np.nan) for r in results]),
             loss=np.array([r.get("loss", np.nan) for r in results]),
             time=np.array([r.get("train_time", r.get("time", np.nan))
                           for r in results]),
             mse_h=np.array([r.get("mse_h", np.nan) for r in results]),
             mse_s=np.array([r.get("mse_s", np.nan) for r in results]),
             mse_F=np.array([r.get("mse_F", np.nan) for r in results]))
    print(f"Comparison NPZ saved to {npz_path}")
    return json_path, npz_path


def print_summary_table(results):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 72)
    print("  COMPARISON SUMMARY")
    print("=" * 72)
    header = f"{'Method':<12} {'E_error':>10} {'Loss':>12} "
    if any("mse_h" in r for r in results):
        header += f"{'MSE(h)':>12} {'MSE(s)':>12} {'MSE(F)':>12} "
    header += f"{'Time(s)':>10} {'Status':>8}"
    print(header)
    print("-" * 72)
    for r in results:
        name = r["method"]
        e_err = f"{r.get('E_error', np.nan):.4f}" if "E_error" in r else "N/A"
        loss = f"{r.get('loss', np.nan):.6e}" if "loss" in r else "N/A"
        row = f"{name:<12} {e_err:>10} {loss:>12}"
        if any("mse_h" in r for r in results):
            mse_h = f"{r.get('mse_h', np.nan):.6e}" if "mse_h" in r else "N/A"
            mse_s = f"{r.get('mse_s', np.nan):.6e}" if "mse_s" in r else "N/A"
            mse_F = f"{r.get('mse_F', np.nan):.6e}" if "mse_F" in r else "N/A"
            row += f" {mse_h:>12} {mse_s:>12} {mse_F:>12}"
        elapsed = r.get("train_time", r.get("time", 0))
        status = "OK" if r.get("success") else "FAIL"
        row += f" {elapsed:>10.1f} {status:>8}"
        print(row)
    print("=" * 72)


def main():
    overall_start = time.perf_counter()

    # ---------- Step 1: target data ----------
    if not args.skip_gen_target:
        if not ensure_target_data():
            print("FATAL: failed to generate target data")
            sys.exit(1)

    results = []

    # ---------- Step 2: baseline ----------
    baseline_result = run_baseline()
    results.append(baseline_result)

    # ---------- Step 3: NN+FD ----------
    nn_fd_result = run_nn_fd()
    results.append(nn_fd_result)

    # ---------- Step 4: NN+AD ----------
    nn_ad_result = run_nn_ad()
    results.append(nn_ad_result)

    # ---------- Step 5: save & summarise ----------
    print_summary_table(results)
    json_path, npz_path = save_results(results)

    overall_elapsed = time.perf_counter() - overall_start
    print(f"\nTotal evaluation time: {overall_elapsed:.1f}s "
          f"({overall_elapsed/60:.1f} min)")

    print(f"\nNext step: run 'python code/plot_comparison.py' to generate the "
          f"comparison matplotlib chart.")

    return 0 if all(r.get("success") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
