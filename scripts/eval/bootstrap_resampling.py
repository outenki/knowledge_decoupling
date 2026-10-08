import numpy as np
import json
from typing import Sequence
import argparse
import os
import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap resampling for SML and CORE performance comparison.")
    parser.add_argument("-sml", "--sml-directory", type=str, required=True, help="Path to the directory containing SML sample JSONL files.")
    parser.add_argument("-core", "--core-directory", type=str, required=True, help="Path to the directory containing CORE sample JSONL files.")
    parser.add_argument("-m", "--metric", type=str, required=True, choices={"acc", "f1", "em"}, help="Metric to use for evaluation.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap resamples. Default is 10000.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for CI. Default is 0.95.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Default is 42.")
    return parser.parse_args()

def get_latest_sample_file_from_directory(directory: str) -> str:
    """
    Get the latest sample JSONL file from a directory.

    Parameters
    ----------
    directory:
        Path to the directory containing sample JSONL files.
    Returns
    -------
    str
        Path to the latest sample JSONL file.
    """

    # Get all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(directory, "samples_*"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in directory: {directory}")

    # Sort files by modification time and get the latest one
    latest_file = max(jsonl_files, key=os.path.getmtime)
    return latest_file

def get_performance(jsonl_file: str, metric: str) -> list:
    """
    Extract sample-level performance values from a JSONL file.

    Parameters
    ----------
    jsonl_file:
        Path to the JSONL file.
    metric:
        Metric to use for evaluation. One of "acc", "f1", "em".
    Returns
    -------
    list
        List of sample-level performance values.
    """
    performance = []
    with open(jsonl_file, "r") as f:
        for line in tqdm(f, desc=f"Processing {jsonl_file}"):
            data = json.loads(line)
            performance.append(data[metric])
    return performance


def bootstrap_ci(
    data: Sequence[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
):
    """
    Bootstrap confidence interval for a single metric.

    Parameters
    ----------
    data:
        Sample-level metric values.
        For example, [0, 1, 1, 0, ...] for accuracy.
    n_bootstrap:
        Number of bootstrap resamples.
    confidence:
        Confidence level, e.g. 0.95.
    seed:
        Random seed.

    Returns
    -------
    dict
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 1:
        raise ValueError("data must be a 1-dimensional array.")

    if len(data) == 0:
        raise ValueError("data must not be empty.")

    rng = np.random.default_rng(seed)
    n = len(data)

    # Original metric
    observed = np.mean(data)

    # Bootstrap resampling
    indices = rng.integers(
        low=0,
        high=n,
        size=(n_bootstrap, n),
    )

    bootstrap_means = np.mean(data[indices], axis=1)

    # Percentile CI
    alpha = 1 - confidence
    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)

    ci = np.percentile(
        bootstrap_means,
        [lower, upper],
    )

    return {
        "n": n,
        "mean": float(observed),
        "bootstrap_mean": float(np.mean(bootstrap_means)),
        "std": float(np.std(data, ddof=1)) if n > 1 else 0.0,
        "ci": [float(ci[0]), float(ci[1])],
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_difference_ci(
    sml: Sequence[float],
    core: Sequence[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
):
    """
    Paired bootstrap confidence interval for CORE - SML.

    sml[i] and core[i] must correspond to the same test sample.
    """

    sml = np.asarray(sml, dtype=float)
    core = np.asarray(core, dtype=float)

    if sml.ndim != 1 or core.ndim != 1:
        raise ValueError("sml and core must be 1-dimensional arrays.")

    if len(sml) != len(core):
        raise ValueError(
            f"sml and core must have the same length: "
            f"{len(sml)} != {len(core)}"
        )

    if len(sml) == 0:
        raise ValueError("sml and core must not be empty.")

    rng = np.random.default_rng(seed)
    n = len(sml)

    # Sample-level paired differences.
    # Positive value means CORE performs better than SML.
    differences = core - sml

    # Observed performance difference
    observed_difference = np.mean(differences)

    # Paired bootstrap:
    # resample test samples, preserving CORE/SML pairing.
    indices = rng.integers(
        low=0,
        high=n,
        size=(n_bootstrap, n),
    )

    bootstrap_differences = np.mean(
        differences[indices],
        axis=1,
    )

    alpha = 1 - confidence
    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)

    ci = np.percentile(
        bootstrap_differences,
        [lower, upper],
    )

    return {
        "n": n,
        "difference": float(observed_difference),
        "direction": "CORE - SML",
        "ci": [float(ci[0]), float(ci[1])],
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def analyze_sml_core(
    sml,
    core,
    n_bootstrap=10000,
    confidence=0.95,
    seed=42,
):
    """
    Perform:

    1. Bootstrap CI for SML.
    2. Bootstrap CI for CORE.
    3. Paired bootstrap CI for CORE - SML.
    """

    sml_result = bootstrap_ci(
        sml,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    core_result = bootstrap_ci(
        core,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed + 1,
    )

    difference_result = bootstrap_difference_ci(
        sml,
        core,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed + 2,
    )

    result = {
        "SML": sml_result,
        "CORE": core_result,
        "CORE_minus_SML": difference_result,
    }

    return result


if __name__ == "__main__":
    args = parse_args()
    sml_jsonl = get_latest_sample_file_from_directory(args.sml_directory)
    print(f"Using latest SML JSONL file: {sml_jsonl}")
    core_jsonl = get_latest_sample_file_from_directory(args.core_directory)
    print(f"Using latest CORE JSONL file: {core_jsonl}")
    sml = get_performance(sml_jsonl, args.metric)
    core = get_performance(core_jsonl, args.metric)

    result = analyze_sml_core(
        sml,
        core,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )

    # save result to output file
    print(f"Saving result to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)

    print(json.dumps(result, indent=4))