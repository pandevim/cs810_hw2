# n_workers from SLURM. Use int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())) so it matches what SLURM actually allocated.

import os
import sys
import random
import glob
import shutil
import subprocess, multiprocessing
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import h5py
import wandb

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

api = HfApi()
HF_REPO_ID = "pandevim/neural_surrogates"
PROJECT_NAME = "hw2"

filenames = ["KS_train_2048.h5", "KS_valid.h5", "KS_test.h5"]
repo_path = f"{PROJECT_NAME}/data/"

def check_data_exists():
    """Returns True if all required files are present locally or on Hugging Face."""
    # Check local data/ directory first — no network call needed
    if all(os.path.exists(f"data/{fn}") for fn in filenames):
        return True

    # Fall back to checking HuggingFace
    try:
        return all(
            api.file_exists(repo_id=HF_REPO_ID, filename=f"{repo_path}{fn}", repo_type="dataset")
            for fn in filenames
        )
    except Exception:
        return False

def download_data():
    """Downloads existing data from Hugging Face cache into the local data directory."""
    print("✓ Data found on HuggingFace. Downloading...")
    os.makedirs("data", exist_ok=True)
    for fn in filenames:
        local_path = f"data/{fn}"
        if os.path.exists(local_path):
            print(f"  ✓ {fn} (already local)")
            continue
        cached_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=f"{repo_path}{fn}", repo_type="dataset"
        )
        shutil.copy(cached_path, local_path)
        print(f"  ✓ {fn}")

def generate_and_upload_data():
    """Generates data from scratch using LPSDA and uploads it to Hugging Face."""
    print("⚠ Data not found. Generating from scratch...")

    # Clone & install
    if not os.path.exists("LPSDA"):
        subprocess.run(
            ["git", "clone", "https://github.com/brandstetter-johannes/LPSDA.git"],
            check=True,
        )

    # Install LPSDA package
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "LPSDA/"], check=True)

    os.makedirs("data", exist_ok=True)
    os.makedirs("LPSDA/data/log", exist_ok=True)
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"Using {n_workers} CPU cores for parallel generation")

    def parallel_generate(total_samples, mode, extra_args=None):
        """Spawns parallel processes to generate trajectories, then merges."""
        samples_per_worker = total_samples // n_workers
        remainder = total_samples % n_workers

        # build the env once, reuse for every worker
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }

        processes = []
        log_files = []

        for i in range(n_workers):
            n = samples_per_worker + (1 if i < remainder else 0)
            if n == 0:
                continue
            cmd = [sys.executable, "generate/generate_data.py",
                  "--experiment=KS", f"--{mode}_samples={n}",
                  f"--train_samples={'0' if mode != 'train' else str(n)}",
                  f"--valid_samples={'0' if mode != 'valid' else str(n)}",
                  f"--test_samples={'0' if mode != 'test' else str(n)}",
                  "--L=64", f"--suffix=part{i}"]
            if extra_args:
                cmd.extend(extra_args)

            log_file = open(f"worker_{mode}_{i}.log", "w")
            log_files.append(log_file)

            processes.append(subprocess.Popen(
                cmd, cwd="LPSDA",
                stdout=log_file, stderr=subprocess.STDOUT,
                env=env,
            ))

        for lf in log_files:
            wandb.save(lf.name, policy="live", base_path=".")

        for j, p in enumerate(processes):
            p.wait()
            log_files[j].close()
            print(f"  Worker {j} finished (exit code {p.returncode})")

    def merge_parts(mode, total_samples, suffix_count, out_filename):
        """Merges partitioned .h5 files into a single file in data/."""
        all_u, all_x, all_dx, all_t, all_dt = [], [], [], [], []

        for i in range(suffix_count):
            # The script names train files differently from valid/test
            if mode == "train":
                # Find the matching file — sample count varies per worker
                candidates = glob.glob(f"LPSDA/data/KS_train_*_part{i}.h5")
            else:
                candidates = glob.glob(f"LPSDA/data/KS_{mode}_part{i}.h5")

            if not candidates:
                print(f"  ⚠ No file found for part{i}, skipping")
                continue

            path = candidates[0]
            with h5py.File(path, "r") as f:
                d = f[mode]
                key = [k for k in d.keys() if k.startswith("pde_")][0]
                all_u.append(d[key][:])
                all_x.append(d["x"][:])
                all_dx.append(d["dx"][:])
                all_t.append(d["t"][:])
                all_dt.append(d["dt"][:])

        u = np.concatenate(all_u)
        print(f"  Merged {u.shape[0]} samples for {mode}")

        with h5py.File(f"data/{out_filename}", "w") as out_f:
            grp = out_f.create_group(mode)
            grp.create_dataset(f"pde_{u.shape[1]}-{u.shape[2]}", data=u)
            grp.create_dataset("x", data=np.concatenate(all_x))
            grp.create_dataset("dx", data=np.concatenate(all_dx))
            grp.create_dataset("t", data=np.concatenate(all_t))
            grp.create_dataset("dt", data=np.concatenate(all_dt))

    # --- Training data (2048 samples, default nt=500) ---
    print("\n[1/3] Generating training data...")
    parallel_generate(2048, "train")
    merge_parts("train", 2048, n_workers, "KS_train_2048.h5")
    wandb.log({"split_complete": "train", "samples": 2048})

    # --- Validation data (128 samples, longer rollout) ---
    print("\n[2/3] Generating validation data...")
    parallel_generate(128, "valid", ["--nt=1000", "--nt_effective=640", "--end_time=200"])
    merge_parts("valid", 128, n_workers, "KS_valid.h5")
    wandb.log({"split_complete": "valid", "samples": 128})

    # --- Test data (128 samples, longer rollout) ---
    print("\n[3/3] Generating test data...")
    parallel_generate(128, "test", ["--nt=1000", "--nt_effective=640", "--end_time=200"])
    merge_parts("test", 128, n_workers, "KS_test.h5")
    wandb.log({"split_complete": "test", "samples": 128})

    # Clean up LPSDA/data parts
    for f in glob.glob("LPSDA/data/KS_*_part*.h5"):
        os.remove(f)
    print("\n✓ Cleaned up temporary part files")

    # Upload
    for filepath in glob.glob("data/KS_*.h5"):
        fn = os.path.basename(filepath)
        print(f"Uploading {fn}...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=f"{repo_path}{fn}",
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        print(f"  ✓ {fn}")
        wandb.log({"uploaded": fn})

if __name__ == "__main__":
    run = wandb.init(
        project="ks-data-generation",
        name=f"slurm-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config={
            "n_workers": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
            "train_samples": 2048,
            "valid_samples": 128,
            "test_samples": 128,
        },
    )

    # Sync the SLURM stdout/stderr files live as they're written
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        for path in glob.glob(f"logs/ani_data_generation_{slurm_job_id}.*"):
            wandb.save(path, policy="live")

    try:
        if check_data_exists():
            download_data()
        else:
            generate_and_upload_data()
    finally:
        wandb.finish() 
