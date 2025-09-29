import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Removed optimization-based planner imports
from pydrake.systems.primitives import VectorLog

# Removed optimization-based planner functions


@dataclass
class PlanarPushingLog:
    t: npt.NDArray[np.float64]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]
    lam: npt.NDArray[np.float64]
    c_n: npt.NDArray[np.float64]
    c_f: npt.NDArray[np.float64]
    lam_dot: npt.NDArray[np.float64]

    @classmethod
    def from_np(
        cls,
        t: npt.NDArray[np.float64],
        state_np_array: npt.NDArray[np.float64],
        control_np_array: npt.NDArray[np.float64],
    ) -> "PlanarPushingLog":
        x = state_np_array[0, :]
        y = state_np_array[1, :]
        theta = state_np_array[2, :]
        if state_np_array.shape[0] == 3:
            # Padding state since we didn't log lam
            lam = np.zeros_like(x)
        else:
            lam = state_np_array[3, :]

        c_n = control_np_array[0, :]
        c_f = control_np_array[1, :]
        lam_dot = control_np_array[2, :]
        return cls(t, x, y, theta, lam, c_n, c_f, lam_dot)

    @classmethod
    def from_log(
        cls,
        state_log: VectorLog,
        control_log: VectorLog,
    ) -> "PlanarPushingLog":
        t = state_log.sample_times()
        state_np_array = state_log.data()
        control_np_array = control_log.data()
        return cls.from_np(t, state_np_array, control_np_array)

    @classmethod
    def from_pose_vector_log(
        cls,
        pose_vector_log: VectorLog,
    ) -> "PlanarPushingLog":
        t = pose_vector_log.sample_times()
        state_np_array = pose_vector_log.data()
        PAD_VAL = 0
        single_row_pad = np.ones_like(state_np_array[0, :]) * PAD_VAL
        if state_np_array.shape[0] == 3:
            # Padding state since we didn't log lam
            state_np_array = np.vstack((state_np_array, single_row_pad))
        elif state_np_array.shape[0] == 2:
            # Padding state since we didn't log theta and lam
            state_np_array = np.vstack(
                (
                    state_np_array,
                    single_row_pad,
                    single_row_pad,
                )
            )
        control_np_array = np.ones((3, len(t))) * PAD_VAL
        return cls.from_np(t, state_np_array, control_np_array)


@dataclass
class CombinedPlanarPushingLogs:
    pusher_actual: PlanarPushingLog
    slider_actual: PlanarPushingLog
    pusher_desired: PlanarPushingLog
    slider_desired: PlanarPushingLog


def plot_planar_pushing_logs(
    state_log: VectorLog,
    state_log_desired: VectorLog,
    control_log: VectorLog,
    control_log_desired: VectorLog,
) -> None:
    actual = PlanarPushingLog.from_log(state_log, control_log)
    desired = PlanarPushingLog.from_log(state_log_desired, control_log_desired)

    plot_planar_pushing_trajectory(actual, desired)


def plot_control_sols_vs_time(control_log: List[np.ndarray], suffix: str = "", save_dir: Optional[str] = None) -> None:
    # Convert the list to a numpy array for easier manipulation
    control_log_array = np.array(control_log)

    # Prepare data for plotting
    timesteps = np.arange(control_log_array.shape[0])
    prediction_horizons = np.arange(control_log_array.shape[1])

    # Create a meshgrid for timesteps and prediction_horizons
    T, P = np.meshgrid(prediction_horizons, timesteps)  # Note the change in the order

    # Initialize a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each control input
    for i, label in enumerate(["c_n", "c_f", "lam_dot"]):
        # Extract the control data for plotting
        Z = control_log_array[:, :, i]

        # Ensure Z has the same shape as T and P
        # Z might need to be transposed depending on how control_log_array is structured
        Z = Z.T if Z.shape != T.shape else Z

        ax.plot_surface(T, P, Z, label=label, alpha=0.7)

    # Adding labels
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Timestep")
    ax.set_zlabel("Control Inputs")

    # Adding title
    ax.set_title("3D Control Inputs Plot")

    # Workaround for legend in 3D plot
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], linestyle="none", marker="_", color="blue", markersize=10),
        Line2D([0], [0], linestyle="none", marker="_", color="orange", markersize=10),
        Line2D([0], [0], linestyle="none", marker="_", color="green", markersize=10),
    ]
    ax.legend(custom_lines, ["c_n", "c_f", "lam_dot"])

    # Show plot
    plt.tight_layout()
    file_name = f"planar_pushing_control_sols{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


def plot_cost(cost_log: List[float], suffix: str = "", save_dir: Optional[str] = None) -> None:
    plt.figure()
    plt.plot(cost_log)
    plt.title("Cost vs. timestep")
    plt.xlabel("timestep")
    plt.ylabel("Cost")
    plt.tight_layout()

    file_name = f"planar_pushing_cost{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


def plot_velocities(
    desired_vel_log: List[npt.NDArray],
    commanded_vel_log: List[npt.NDArray],
    suffix: str = "",
) -> None:
    plt.figure()
    # velocity has x and y dimensions
    desired_vel_log_array = np.array(desired_vel_log)
    commanded_vel_log_array = np.array(commanded_vel_log)
    timesteps = np.arange(desired_vel_log_array.shape[0])
    plt.plot(timesteps, desired_vel_log_array[:, 0], label="desired x vel")
    plt.plot(timesteps, desired_vel_log_array[:, 1], label="desired y vel")
    plt.plot(timesteps, commanded_vel_log_array[:, 0], label="commanded x vel")
    plt.plot(timesteps, commanded_vel_log_array[:, 1], label="commanded y vel")
    plt.legend()
    plt.title("Desired and commanded velocities vs. timestep")
    plt.xlabel("timestep")
    plt.ylabel("Velocity")
    plt.tight_layout()
    plt.savefig(f"planar_pushing_velocities{suffix}.png")


def plot_and_save_planar_pushing_logs_from_sim(
    pusher_pose_vector_log: VectorLog,
    slider_pose_vector_log: VectorLog,
    control_log: VectorLog,
    control_desired_log: VectorLog,
    pusher_pose_vector_log_desired: VectorLog,
    slider_pose_vector_log_desired: VectorLog,
    save_dir: Optional[str] = None,
) -> None:
    pusher_actual = PlanarPushingLog.from_pose_vector_log(pusher_pose_vector_log)
    slider_actual = PlanarPushingLog.from_log(slider_pose_vector_log, control_log)
    pusher_desired = PlanarPushingLog.from_pose_vector_log(pusher_pose_vector_log_desired)
    slider_desired = PlanarPushingLog.from_log(
        slider_pose_vector_log_desired,
        control_desired_log,
    )
    # Save the logs
    combined = CombinedPlanarPushingLogs(
        pusher_actual=pusher_actual,
        slider_actual=slider_actual,
        pusher_desired=pusher_desired,
        slider_desired=slider_desired,
    )

    with open(f"{save_dir}/combined_planar_pushing_logs.pkl", "wb") as f:
        pickle.dump(combined, f)

    plot_planar_pushing_trajectory(
        slider_actual,
        slider_desired,
        suffix="_slider",
        plot_lam=False,
        save_dir=save_dir,
    )
    plot_planar_pushing_trajectory(
        pusher_actual,
        pusher_desired,
        suffix="_pusher",
        plot_lam=False,
        plot_control=False,
        save_dir=save_dir,
    )


def plot_planar_pushing_trajectory(
    actual: PlanarPushingLog,
    desired: PlanarPushingLog,
    plot_control: bool = True,
    plot_lam: bool = True,
    suffix: str = "",
    save_dir: Optional[str] = None,
) -> None:
    # State plot
    fig, axes = plt.subplots(nrows=4 if plot_lam else 3, ncols=1, figsize=(8, 8))
    MIN_AXIS_SIZE = 0.1

    pos = np.vstack((actual.x, actual.y))
    max_pos_change = max(np.ptp(np.linalg.norm(pos, axis=0)), MIN_AXIS_SIZE) * 1.3
    # Note: this calculation doesn't center the plot on the right value, so
    # the line might not be visible

    axes[0].plot(actual.t, actual.x, label="Actual")
    axes[0].plot(actual.t, desired.x, linestyle="--", label="Desired")
    axes[0].set_title("x")
    axes[0].legend()
    # axes[0].set_ylim(-max_pos_change, max_pos_change)

    axes[1].plot(actual.t, actual.y, label="Actual")
    axes[1].plot(actual.t, desired.y, linestyle="--", label="Desired")
    axes[1].set_title("y")
    axes[1].legend()
    axes[1].set_ylim(-max_pos_change, max_pos_change)

    th_change = max(np.ptp(actual.theta), MIN_AXIS_SIZE) * 2.0  # type: ignore

    axes[2].plot(actual.t, actual.theta, label="Actual")
    axes[2].plot(actual.t, desired.theta, linestyle="--", label="Desired")
    axes[2].set_title("theta")
    axes[2].legend()
    # axes[2].set_ylim(-th_change, th_change)

    if plot_lam:
        axes[3].plot(actual.t, actual.lam, label="Actual")
        axes[3].plot(actual.t, desired.lam, linestyle="--", label="Desired")
        axes[3].set_title("lam")
        axes[3].legend()
        axes[3].set_ylim(0, 1)

    plt.tight_layout()
    file_name = f"planar_pushing_states{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)

    # State Error plot
    fig, axes = plt.subplots(nrows=4 if plot_lam else 3, ncols=1, figsize=(8, 8))
    MIN_AXIS_SIZE = 0.1

    x_error = actual.x - desired.x
    y_error = actual.y - desired.y
    pos = np.vstack((x_error, y_error))
    # max_pos_change = max(np.ptp(np.linalg.norm(pos, axis=0)), MIN_AXIS_SIZE) * 1.3
    max_pos_change = 0.1

    axes[0].plot(actual.t, x_error, label="Error")
    axes[0].set_title("x")
    axes[0].plot(actual.t, np.zeros_like(actual.t), linestyle="--", label="0")
    axes[0].legend()
    axes[0].set_ylim(-max_pos_change, max_pos_change)

    axes[1].plot(actual.t, y_error, label="Error")
    axes[1].plot(actual.t, np.zeros_like(actual.t), linestyle="--", label="0")
    axes[1].set_title("y")
    axes[1].legend()
    axes[1].set_ylim(-max_pos_change, max_pos_change)

    theta_error = actual.theta - desired.theta
    th_change = max(np.ptp(theta_error), MIN_AXIS_SIZE) * 2.0  # type: ignore
    th_change = 0.1

    axes[2].plot(actual.t, theta_error, label="Error")
    axes[2].set_title("theta")
    axes[2].plot(actual.t, np.zeros_like(actual.t), linestyle="--", label="0")
    axes[2].legend()
    axes[2].set_ylim(-th_change, th_change)

    if plot_lam:
        lam_error = actual.lam - desired.lam
        axes[3].plot(actual.t, lam_error, label="Error")
        axes[3].plot(actual.t, np.zeros_like(actual.t), linestyle="--", label="0")
        axes[3].set_title("lam")
        axes[3].legend()
        axes[3].set_ylim(0, 1)

    plt.tight_layout()
    file_name = f"planar_pushing_states_error{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)

    if not plot_control:
        return

    # Control input
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))

    if len(actual.c_n) == len(actual.t):
        pass
    elif len(actual.c_n) == len(actual.t) - 1:
        actual.t = actual.t[:-1]
    else:
        raise ValueError("Mismatch in data length")

    max_force_change = max(max(np.ptp(actual.c_n), np.ptp(actual.c_f)), MIN_AXIS_SIZE) * 2  # type: ignore

    # Plot lines on each subplot
    axes[0].plot(actual.t, actual.c_n, label="Actual")
    axes[0].plot(actual.t, desired.c_n, linestyle="--", label="Desired")
    axes[0].set_title("c_n")
    axes[0].legend()
    # axes[0].set_ylim(-max_force_change, max_force_change)

    axes[1].plot(actual.t, actual.c_f, label="Actual")
    axes[1].plot(actual.t, desired.c_f, linestyle="--", label="Desired")
    axes[1].set_title("c_f")
    axes[1].legend()
    # axes[1].set_ylim(-max_force_change, max_force_change)

    max_lam_dot_change = max(np.ptp(actual.lam_dot), MIN_AXIS_SIZE) * 1.3  # type: ignore
    axes[2].plot(actual.t, actual.lam_dot, label="Actual")
    axes[2].plot(actual.t, desired.lam_dot, linestyle="--", label="Desired")
    axes[2].set_title("lam_dot")
    axes[2].legend()
    # axes[2].set_ylim(-max_lam_dot_change, max_lam_dot_change)

    # Adjust layout
    plt.tight_layout()
    file_name = f"planar_pushing_control{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


def plot_realtime_rate(
    real_time_rate_log: List[float],
    time_step: float,
    suffix: str = "",
    save_dir: Optional[str] = None,
) -> None:
    plt.figure()
    plt.plot(real_time_rate_log)
    plt.title("Realtime rate vs. timestep")
    plt.xticks(np.arange(0, len(real_time_rate_log), 1 / time_step), rotation=90)
    plt.xlabel("timestep")
    plt.ylabel("Realtime rate")
    # Add grid
    plt.grid()
    plt.tight_layout()
    file_name = f"planar_pushing_realtime_rate{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


def plot_mpc_solve_times(
    solve_times_log: Dict[str, List[float]],
    suffix: str = "",
    save_dir: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots()
    for key, solve_times in solve_times_log.items():
        ax.plot(solve_times, label=key)
    ax.set_title("MPC solve times vs. timestep")
    ax.set_xlabel("timestep")
    ax.set_ylabel("Solve times")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    file_name = f"planar_pushing_mpc_solve_times{suffix}.png"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


def plot_joint_state_logs(joint_state_log, num_positions, suffix="", save_dir=""):
    num_velocities = joint_state_log.data().shape[0] - num_positions
    # Split the data into positions and velocities
    data = joint_state_log.data()
    sample_times = joint_state_log.sample_times()
    positions = data[:num_positions, :]
    velocities = data[num_positions:, :]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plotting positions
    for i in range(num_positions):
        axs[0].plot(sample_times, positions[i, :], label=f"Joint {i + 1}")
    axs[0].set_title("Joint Positions")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    # Plotting velocities
    for i in range(num_velocities):
        axs[1].plot(sample_times, velocities[i, :], label=f"Joint {i + 1}")
    axs[1].set_title("Joint Velocities")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    file_name = f"joint_states{suffix}.pdf"
    file_path = f"{save_dir}/{file_name}" if save_dir else file_name
    plt.savefig(file_path)


PLOT_WIDTH_INCH = 7
PLOT_HEIGHT_INCH = 4.5

DISTANCE_REF = 0.3  # Width of box
FORCE_REF = 10  # Current max force
TORQUE_REF = FORCE_REF * DISTANCE_REF  # Almost max torque


# Removed optimization-based planner analysis functions


def show_plots() -> None:
    plt.show()


def _create_curve_norm(
    curve: npt.NDArray[np.float64],  # (N, dims)
) -> npt.NDArray[np.float64]:  # (N, 1)
    return np.apply_along_axis(np.linalg.norm, 1, curve).reshape((-1, 1))


# Removed optimization-based planner analysis functions


# Removed optimization-based planner constraint analysis functions
