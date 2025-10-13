import argparse
import csv
import os
import pickle
import shlex
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta

# Common arguments
CONFIG_DIR = "config/sim_config/sim_sim"
CONFIG_NAME = "gamepad_teleop.yaml"
BASE_COMMAND = [
    "python",
    "scripts/run_sim_sim_eval.py",
    f"--config-dir={CONFIG_DIR}",
]

# ---------------------------------------------------------
# Example Usage:
# python launch_evals.py --csv-path /path/to/jobs.csv --max-concurrent-jobs-per-gpu 8
#
# CSV file format:
# checkpoint_path,run_dir,config_name (optional),overrides (optional)
# /path/to/checkpoint1.ckpt, data/test1, custom_config.yaml, diffusion_policy_config.cfg_overrides.n_action_steps=4
# /path/to/checkpoint2.ckpt, data/test2, , "param1=value1 param2=value2"
# ---------------------------------------------------------


@dataclass
class JobConfig:
    checkpoint_path: str
    run_dir: str
    config_name: str
    num_trials: int = -1
    seed: int = 0
    continue_flag: bool = False
    group_key: str = ""
    overrides: str = ""  # Hydra config overrides (space-separated)
    gpu_id: int = 0

    def __str__(self):
        return (
            f"checkpoint_path={self.checkpoint_path}, "
            f"run_dir={self.run_dir}, "
            f"config_name={self.config_name}, "
            f"num_trials={self.num_trials}, "
            f"seed={self.seed}, "
            f"continue_flag={self.continue_flag}, "
            f"group_key={self.group_key}, "
            f"overrides={self.overrides}"
        )

    def __repr__(self):
        return str(self)


@dataclass
class JobResult:
    num_successful_trials: int
    num_trials: int
    job_config: JobConfig = None

    def __post_init__(self):
        self.success_rate = self.num_successful_trials / self.num_trials

    def __str__(self):
        return (
            f"num_successful_trials={self.num_successful_trials}, "
            f"num_trials={self.num_trials}, "
            f"success_rate={self.success_rate}"
        )

    def __repr__(self):
        return str(self)


class GPUQueue:
    """
    Queue manager for distributing jobs across GPUs. Simply ensures no GPU is assigned more than max_jobs_per_gpu
    concurrent jobs.
    """

    def __init__(self, num_gpus, max_jobs_per_gpu):
        self.num_gpus = num_gpus
        self.max_jobs_per_gpu = max_jobs_per_gpu
        self.gpu_job_counts = defaultdict(int)
        self.lock = threading.Lock()

    def get_next_gpu(self):
        """Get the next available GPU, or None if all are full."""
        with self.lock:
            for gpu_id in range(self.num_gpus):
                if self.gpu_job_counts[gpu_id] < self.max_jobs_per_gpu:
                    self.gpu_job_counts[gpu_id] += 1
                    return gpu_id
            return None

    def release_gpu(self, gpu_id):
        """Release a GPU slot."""
        with self.lock:
            self.gpu_job_counts[gpu_id] = max(0, self.gpu_job_counts[gpu_id] - 1)

    def get_total_max_jobs(self):
        """Get total maximum concurrent jobs across all GPUs."""
        return self.num_gpus * self.max_jobs_per_gpu


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch multiple Hydra simulation commands concurrently.")
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the CSV file containing checkpoint paths, run directories, and optional config names.",
    )
    parser.add_argument(
        "--max-concurrent-jobs-per-gpu",
        type=int,
        default=4,
        help="Maximum number of concurrent jobs per GPU (default: 4).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for the evaluation (default: 1).",
    )
    parser.add_argument(
        "--num-trials-per-round",
        type=int,
        nargs="+",
        default=[50, 50, 100],
        help="List of number of trials per round (default: [50, 50, 100]).",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.05,
        help="Threshold for dropping checkpoints (default: 0.05).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts and automatically overwrite existing output directories.",
    )
    return parser.parse_args()


def get_checkpoint_root_and_name(checkpoint_path):
    """Get the root directory and name of the checkpoint."""
    checkpoint_root = checkpoint_path.split("/checkpoints")[0]
    checkpoint_name = checkpoint_path.split("/")[-1]
    return checkpoint_root, checkpoint_name


def _make_unique_group_key(base_key, existing_groups):
    """Return a stable key that differentiates duplicate training runs."""
    if base_key not in existing_groups:
        return base_key

    suffix = 2
    while True:
        candidate = f"{base_key} (entry {suffix})"
        if candidate not in existing_groups:
            return candidate
        suffix += 1


def load_jobs_from_csv(csv_file):
    """Load checkpoint groups, where each group consists of one or more checkpoints."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

    job_groups = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoint_path = row.get("checkpoint_path", "").strip()
            run_dir = row.get("run_dir", "").strip()
            config_name = row.get("config_name", CONFIG_NAME).strip()
            overrides = row.get("overrides", "").strip()

            # If evaluating a single checkpoint, create a single-element group
            if checkpoint_path.endswith(".ckpt"):
                assert os.path.exists(checkpoint_path), f"Checkpoint file '{checkpoint_path}' does not exist."
                checkpoint_root, checkpoint_file = get_checkpoint_root_and_name(checkpoint_path)
                group_key = _make_unique_group_key(checkpoint_root, job_groups)

                job_config = JobConfig(
                    checkpoint_path=checkpoint_path,
                    run_dir=f"{run_dir}/{checkpoint_file}",
                    config_name=config_name,
                    seed=0,
                    continue_flag=False,
                    group_key=group_key,
                    overrides=overrides,
                    gpu_id=0,  # Default to 0 for single checkpoint evaluation
                )
                job_groups[group_key] = {checkpoint_file: job_config}

            # If evaluating all checkpoints from a training run, create a group
            else:
                checkpoint_group = {}
                checkpoints_dir = os.path.join(checkpoint_path, "checkpoints")
                group_key = _make_unique_group_key(checkpoint_path, job_groups)
                for checkpoint_file in os.listdir(checkpoints_dir):
                    if checkpoint_file.endswith(".ckpt"):
                        full_checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
                        _, checkpoint_file = get_checkpoint_root_and_name(full_checkpoint_path)
                        job_config = JobConfig(
                            checkpoint_path=full_checkpoint_path,
                            run_dir=os.path.join(run_dir, checkpoint_file),
                            config_name=config_name,
                            seed=0,
                            continue_flag=False,
                            group_key=group_key,
                            overrides=overrides,
                            gpu_id=0,  # Default to 0 for single checkpoint evaluation
                        )
                        checkpoint_group[checkpoint_file] = job_config
                job_groups[group_key] = checkpoint_group

    return job_groups


# A global lock to protect access to the shared `used_ports` set.
_meshcat_port_lock = threading.Lock()


def get_next_free_meshcat_port(start_port: int = 7000) -> int:
    """Return an available Meshcat port number.

    This function is now thread-safe. Concurrent invocations from different
    worker threads can no longer allocate the same port at the same time.
    """
    if not hasattr(get_next_free_meshcat_port, "used_ports"):
        # We store the set on the function object to keep the global footprint
        # minimal while still sharing state between calls.
        get_next_free_meshcat_port.used_ports = set()

    with _meshcat_port_lock:
        next_port = start_port
        while next_port in get_next_free_meshcat_port.used_ports:
            next_port += 1

        get_next_free_meshcat_port.used_ports.add(next_port)
        return next_port


def free_meshcat_port(meshcat_port):
    """Release a previously reserved Meshcat port."""
    if not hasattr(get_next_free_meshcat_port, "used_ports"):
        return  # Nothing to free

    with _meshcat_port_lock:
        get_next_free_meshcat_port.used_ports.discard(meshcat_port)


def run_simulation(job_config, job_number, total_jobs, round_number, total_rounds, gpu_queue):
    """Run a single simulation with specified checkpoint, run directory, and config name."""
    checkpoint_path = job_config.checkpoint_path
    run_dir = job_config.run_dir
    config_name = job_config.config_name
    num_trials = job_config.num_trials
    seed = job_config.seed
    continue_flag = job_config.continue_flag
    overrides = job_config.overrides
    assert num_trials > 0, "num_trials must be greater than 0"

    meshcat_port = get_next_free_meshcat_port()

    # Get GPU assignment from queue
    gpu_id = gpu_queue.get_next_gpu()
    if gpu_id is None:
        print(f"No GPU available for job {job_number}, this shouldn't happen with proper queue management")
        return None

    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # hide the other GPU

        command = BASE_COMMAND + [
            f"--config-name={config_name}",
            f'diffusion_policy_config.checkpoint="{checkpoint_path}"',
            f'hydra.run.dir="{run_dir}"',
            f"multi_run_config.seed={seed}",
            f"multi_run_config.num_runs={num_trials}",
            f"meshcat_port={meshcat_port}",
            f"++continue_eval={continue_flag}",
            "diffusion_policy_config.device=cuda:0",  # 0 in the *local* namespace
        ]

        # Add any custom overrides from CSV
        if overrides:
            # Split by spaces but respect quoted strings
            override_list = shlex.split(overrides)
            command.extend(override_list)

        # Format command string with each option on a new line for readability
        command_str = " \\\n    ".join(command)

        print("\n" + "=" * 100)
        print(f"=== Round ({round_number} of {total_rounds}): JOB {job_number} of {total_jobs} ===")
        print(f"=== JOB START: {run_dir} ===")
        print(f"=== GPU: {gpu_id} ===")
        print(command_str)
        print(f"Meshcat URL: http://localhost:{meshcat_port}")
        print("=" * 100 + "\n")

        result = subprocess.run(command, capture_output=True, text=True, env=env)

        if result.returncode == 0:
            print(f"\n✅ Completed: {run_dir} (GPU {gpu_id})")

            # Compute success rate
            summary_file = os.path.join(run_dir, "summary.pkl")
            with open(summary_file, "rb") as f:
                summary = pickle.load(f)
            num_successful_trials = len(summary["successful_trials"])
            num_trials = len(summary["trial_times"])
            success_rate = num_successful_trials / num_trials
        else:
            print(f"\n❌ Failed: {run_dir} (GPU {gpu_id})\nError: {result.stderr}")
            success_rate = None

        print("\n" + "=" * 50)
        print(f"=== JOB END: {run_dir} (GPU {gpu_id}) ===")
        if success_rate is not None:
            print(f"Success Rate: {success_rate:.6f} ({num_successful_trials}/{num_trials})")
        else:
            print("Success Rate: None")
        print("=" * 50 + "\n")

        if success_rate is None:
            return None
        else:
            return JobResult(
                num_successful_trials=len(summary["successful_trials"]),
                num_trials=len(summary["trial_times"]),
                job_config=job_config,
            )

    finally:
        # Always release the GPU slot and meshcat port
        gpu_queue.release_gpu(gpu_id)
        free_meshcat_port(meshcat_port)


def validate_job_groups(job_groups, force=False):
    if not job_groups:
        print("No valid jobs found in the CSV file. Please check the file.")
        return False

    # Sure there are no duplicate logging directories in the jobs list
    logging_dirs = []
    for _, group in job_groups.items():
        for _, job in group.items():
            logging_dirs.append(job.run_dir)
    if len(logging_dirs) != len(set(logging_dirs)):
        print("Duplicate logging directories found in the jobs list.")
        return False

    # Double check if output directories already exist
    for _, group in job_groups.items():
        for _, job in group.items():
            output_dir = job.run_dir
            if os.path.exists(output_dir):
                print(f"Output dir '{output_dir}' already exists. Running this job will delete the existing contents.")
                if force:
                    print("--force flag set. Deleting output directory...\n")
                    shutil.rmtree(output_dir)
                else:
                    resp = input("Run job anyways? [y/n]: ")
                    if resp.lower() == "y":
                        print("Deleting output directory...\n")
                        shutil.rmtree(output_dir)
                    else:
                        print("Exiting...")
                        return False

    return True


def print_diagnostic_info(job_groups, max_concurrent_jobs_per_gpu, num_gpus, num_trials, drop_threshold):
    num_jobs = sum([len(group) for group in job_groups.values()])
    total_max_jobs = num_gpus * max_concurrent_jobs_per_gpu

    print("\nDiagnostic Information:")
    print("=======================")
    print(f"Evaluating {len(job_groups)} training runs, consisting of {num_jobs} checkpoints")
    print(f"Using {num_gpus} GPU(s) with {max_concurrent_jobs_per_gpu} jobs per GPU")
    print(f"Total maximum concurrent jobs: {total_max_jobs}")
    print(f"Checkpoints will be compared at {num_trials} trials.")
    print(
        f"During each comparison, if the probability that a checkpoint "
        f"is better than the current best checkpoint is less than {drop_threshold}, "
        f"the checkpoint will be dropped."
    )
    print(f"The best checkpoints will be evaluated for {sum(num_trials)} trials.")
    print("\nTraining run details:")

    for training_dir, group in job_groups.items():
        print("------------------------------")
        print(f"Training Run: {training_dir}")
        print("Checkpoints:")
        for i, job_item in enumerate(group.items()):
            _, job = job_item
            print(f"  {i + 1}. {os.path.basename(job.checkpoint_path)}")
        print(f"Eval directory: {job.run_dir}")
        print(f"Config Name: {job.config_name}")
        if job.overrides:
            print(f"Overrides: {job.overrides}")
        print()
    print()


def prob_p1_greater_p2(n1, N1, n2, N2):
    """
    Computes P(p1 > p2) where:
    - n1, N1: Successes and trials for p1
    - n2, N2: Successes and trials for p2

    Returns:
    - Probability that p1 > p2
    """

    # Numerical integration
    alpha1, beta1 = n1 + 1, N1 - n1 + 1
    alpha2, beta2 = n2 + 1, N2 - n2 + 1

    def cdf_p1(x):
        return beta.cdf(x, alpha1, beta1)

    def pdf_p2(x):
        return beta.pdf(x, alpha2, beta2)

    # p(p1 > p2) = int_0^1 cdf_p1(x) * pdf_p2(x) dx
    integral, _ = quad(lambda x: (1 - cdf_p1(x)) * pdf_p2(x), 0, 1)
    return integral


def determine_new_jobs_to_run(success_rates, drop_threshold):
    jobs_to_run = []
    for group, completed_jobs in success_rates.items():
        # Check for Nones
        has_none = False
        for checkpoint, result in completed_jobs.items():
            if result is None:
                has_none = True
                break
        if has_none:
            print(f"Skipping group {group} due to None values.")
            continue

        # Check for only one job
        if len(completed_jobs) == 1:
            result = list(completed_jobs.values())[0]
            jobs_to_run.append(result.job_config)
            continue

        # Find the best job
        best_job_success_rate = 0
        best_job_num_successful_trials = 0
        best_job_num_trials = 0
        for checkpoint, result in completed_jobs.items():
            if result.success_rate > best_job_success_rate:
                best_job_success_rate = result.success_rate
                best_job_num_successful_trials = result.num_successful_trials
                best_job_num_trials = result.num_trials

        # Compare all other jobs to the best job
        for checkpoint, result in completed_jobs.items():
            if result.success_rate == best_job_success_rate:
                jobs_to_run.append(result.job_config)
                continue

            # Compare the job to the best job
            prob = prob_p1_greater_p2(
                result.num_successful_trials,
                result.num_trials,
                best_job_num_successful_trials,
                best_job_num_trials,
            )
            if prob < drop_threshold:
                print(
                    f"Dropping {checkpoint} with success rate {result.success_rate} from group {group}."
                    f"(p(ckpt > best) = {prob:.6f})"
                )
            else:
                jobs_to_run.append(result.job_config)

    return jobs_to_run


def print_best_checkpoints(success_rates, job_groups):
    print("Final Results (Best Checkpoints):")
    print("=======================")
    for group in job_groups.keys():
        if len(success_rates[group]) == 0:
            print(f"{group}:\n  error (please rerun)\n")
            continue

        # if group has None result, a job has failed along the way
        has_none = False
        for checkpoint, result in success_rates[group].items():
            if result is None:
                has_none = True
                break
        if has_none:
            print(f"{group}:\n  error (please rerun)\n")
            continue

        # find the best job
        best_result = JobResult(0, 1)  # success rate of 0
        for checkpoint, result in success_rates[group].items():
            if result.success_rate > best_result.success_rate:
                best_result = result
            elif result.success_rate == best_result.success_rate:
                _, checkpoint_file = get_checkpoint_root_and_name(result.job_config.checkpoint_path)
        print(f"{group}:")
        for checkpoint, result in success_rates[group].items():
            if result.success_rate == best_result.success_rate:
                _, checkpoint_file = get_checkpoint_root_and_name(result.job_config.checkpoint_path)
                print(
                    f"{checkpoint_file}: {result.success_rate:.6f} ({result.num_successful_trials}/{result.num_trials})"
                )
        print()


def main():
    args = parse_arguments()
    csv_file = args.csv_path
    max_concurrent_jobs_per_gpu = args.max_concurrent_jobs_per_gpu
    num_gpus = args.num_gpus
    num_trials_per_round = args.num_trials_per_round  # default: [50, 50, 100]
    drop_threshold = args.drop_threshold

    # Create GPU queue manager
    gpu_queue = GPUQueue(num_gpus, max_concurrent_jobs_per_gpu)

    # 1 job group per line in csv file; each group contains all jobs corresponding to a single checkpoint
    job_groups = load_jobs_from_csv(csv_file)
    if not validate_job_groups(job_groups, args.force):
        return

    print_diagnostic_info(job_groups, max_concurrent_jobs_per_gpu, num_gpus, num_trials_per_round, drop_threshold)

    # Flatten all job configs from all groups into a single list
    jobs_to_run = [job_config for group in job_groups.values() for job_config in group.values()]

    # Execute multiple rounds of evaluation, with increasing trial counts, dropping poor checkpoints after each round
    # Iterate through rounds
    for i, num_trails_in_round_i in enumerate(num_trials_per_round):
        round_number = i + 1

        # Initialize tracking structures for every job group for this round
        success_rates = {group: {} for group in job_groups.keys()}  # Track success rates by group and checkpoint
        num_jobs_per_group = {group: 0 for group in job_groups.keys()}  # Count jobs per group

        # Count how many jobs will run in each group for this round
        for job_config in jobs_to_run:
            num_jobs_per_group[job_config.group_key] += 1

        # Print total number of jobs and jobs per checkpoint
        print(f"\nRound {i + 1} of {len(num_trials_per_round)}: Running {len(jobs_to_run)} jobs:")
        for group, num_jobs in num_jobs_per_group.items():
            print(f"  {group}: {num_jobs} job(s)")

        # Update job configurations for this round
        for job_config in jobs_to_run:
            job_config.num_trials = num_trails_in_round_i  # Set number of trials for this round
            job_config.seed = i  # Set seed to round number for reproducibility
            job_config.continue_flag = i != 0  # Continue from previous results if not first round

        # Execute jobs concurrently using ThreadPoolExecutor
        total_max_jobs = gpu_queue.get_total_max_jobs()
        with ThreadPoolExecutor(max_workers=total_max_jobs) as executor:
            # Submit all jobs to the thread pool
            futures = {}
            # Iterate through jobs (which may consist of 50 or 100 trials) per round
            # Each job corresponds to 1 checkpoint from 1 line in the csv file
            # (i.e. a CSV file with 3 lines and 10 checkpoints per line --> 30 jobs)
            for job_number, job_config in enumerate(jobs_to_run):
                # Submit each job to the executor
                future = executor.submit(
                    run_simulation,
                    job_config,
                    job_number + 1,
                    len(jobs_to_run),
                    round_number,
                    len(num_trials_per_round),
                    gpu_queue,
                )
                futures[future] = job_config
                time.sleep(0.1)  # So that prints don't overlap

            # Wait for all jobs to complete and collect results
            for future in as_completed(futures):
                job_result = future.result()

                if job_result is not None:
                    job_config = job_result.job_config
                else:  # Failed job
                    job_config = futures[future]

                group = job_config.group_key
                _, checkpoint_file = get_checkpoint_root_and_name(job_config.checkpoint_path)

                # Ensure no duplicate checkpoint names within a group
                assert (
                    checkpoint_file not in success_rates[group]
                ), f"Duplicate checkpoint {checkpoint_file} in group {group}"

                # Store the result for this checkpoint
                success_rates[group][checkpoint_file] = job_result

        # After each round (except the last), determine which jobs to keep for the next round
        # This implements a multi-round elimination strategy based on statistical significance
        if i != len(num_trials_per_round) - 1:
            jobs_to_run = determine_new_jobs_to_run(success_rates, drop_threshold)

    print("\n✅ All jobs finished.\n")
    print_best_checkpoints(success_rates, job_groups)


def create_probability_grid(N1, N2, threshold=0.05):
    grid = np.zeros((N1 + 1, N2 + 1), dtype=bool)

    for n1 in range(N1 + 1):
        for n2 in range(N2 + 1):
            prob = prob_p1_greater_p2(n1, N1, n2, N2)
            grid[n1, n2] = prob < threshold

    return grid


def visualize_grid(grid, N1, N2, threshold):
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap="coolwarm", origin="lower", extent=(0, N2, 0, N1))
    plt.colorbar(label=f"Probability < {threshold}")
    plt.title(f"Grid of P(p1 > p2) < {threshold}")
    plt.xlabel("n2 (better policy)")
    plt.ylabel("n1 (worse policyt)")
    plt.xticks(np.arange(0, N2 + 1, step=max(1, N2 // 10)))
    plt.yticks(np.arange(0, N1 + 1, step=max(1, N1 // 10)))

    # Annotate the grid with 'T' and 'F'
    for i in range(N1 + 1):
        for j in range(N2 + 1):
            text = "T" if grid[i, j] else "F"
            plt.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig("probability_grid.png")


# if __name__ == "__main__":
#     # Parameters
#     N1 = 50
#     N2 = 50
#     threshold = 0.03

#     # Create and visualize the grid
#     grid = create_probability_grid(N1, N2, threshold)
#     visualize_grid(grid, N1, N2, threshold)

if __name__ == "__main__":
    main()
