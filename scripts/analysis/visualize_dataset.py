import argparse
import os

import cv2
import numpy as np
import zarr


def play_trajectory_videos(zarr_path, fps, stride=1, image_size=None, output_path=None, playback_speed=1.0):
    """
    Play trajectory videos from the overhead camera data in the Zarr dataset.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        fps (int): Frames per second for the video.
        stride (int): Number of episodes to skip when playing videos.
        image_size (tuple): Desired image size (width, height) for resizing. None for original size.
        output_path (str): Path to save videos. If None, only display.
            If provided, saves to output_path/trajectory_N.mp4
        playback_speed (float): Speed multiplier for playback/saving. 1.0 = normal, 2.0 = 2x speed, 0.5 = half speed.
    """
    # Load Zarr dataset
    dataset = zarr.open(zarr_path, mode="r")

    # Extract overhead camera data and episode ends
    overhead_camera = dataset["data"]["overhead_camera"]
    episode_ends = dataset["meta"]["episode_ends"][:]

    # Compute start indices for each episode
    episode_starts = [0] + episode_ends[:-1].tolist()

    # Create output directory if saving videos
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    for i in range(0, len(episode_starts), stride):
        start_idx = episode_starts[i]
        end_idx = episode_ends[i]

        # Extract images for the current episode
        trajectory_images = overhead_camera[start_idx:end_idx]

        # Convert images to BGR (for opencv)
        trajectory_images = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in trajectory_images])

        # Resize images if image_size is specified
        if image_size is not None:
            trajectory_images = np.array([cv2.resize(img, image_size) for img in trajectory_images])

        # Save video if output_path is provided
        if output_path is not None:
            video_file = os.path.join(output_path, f"trajectory_{i}.mp4")
            height, width = trajectory_images[0].shape[:2]
            # Apply playback speed to video fps
            adjusted_fps = fps * playback_speed
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_file, fourcc, adjusted_fps, (width, height))
            for img in trajectory_images:
                video_writer.write(img)
            video_writer.release()
            print(f"Saved video to {video_file} (playback speed: {playback_speed}x)")

        # Display the video (skip if only saving)
        if output_path is None:
            # Apply playback speed to display delay
            delay_ms = int(1000 / (fps * playback_speed))
            for img in trajectory_images:
                cv2.imshow(f"Trajectory {i + 1}", img)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    print("Video playback interrupted by user.")
                    return

            # Close the window after the trajectory video ends
            cv2.destroyWindow(f"Trajectory {i + 1}")


if __name__ == "__main__":
    """
    Example usage:
    # Display videos interactively
    python scripts/analysis/visualize_dataset.py \
        --zarr-path ~/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_tee_data.zarr \
        --fps 30 \
        --stride 1 \
        --image-size 640x480

    # Save videos to disk at 2x speed
    python scripts/analysis/visualize_dataset.py \
        --zarr-path ~/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_tee_data.zarr \
        --fps 30 \
        --stride 1 \
        --image-size 640x480 \
        --output-path ./videos \
        --playback-speed 0.5
    """
    parser = argparse.ArgumentParser(description="Play trajectory videos from a Zarr dataset.")
    parser.add_argument("--zarr-path", type=str, required=True, help="Path to the Zarr dataset.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Number of episodes to skip between videos.",
    )
    parser.add_argument(
        "--image-size",
        type=lambda s: tuple(map(int, s.split("x"))),
        default=None,
        help="Desired image size in the format WIDTHxHEIGHT (e.g., 640x480).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Directory to save videos. If not provided, videos are only displayed.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 1.0 = normal, 2.0 = 2x speed, 0.5 = half speed.",
    )
    args = parser.parse_args()

    play_trajectory_videos(
        zarr_path=args.zarr_path,
        fps=args.fps,
        stride=args.stride,
        image_size=args.image_size,
        output_path=args.output_path,
        playback_speed=args.playback_speed,
    )
