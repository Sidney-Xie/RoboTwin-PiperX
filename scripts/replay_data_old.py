#!/usr/bin/env python3
"""Replay ``data_old`` with calibrated cameras and measured gripper states.

The saved successful seeds and arm trajectories are authoritative.  Each
episode is reconstructed with the current embodiment/camera configuration,
then the following HDF5 datasets are replaced atomically:

* ``vision/{cam_head,cam_left_wrist,cam_right_wrist}`` colors, intrinsics,
  extrinsics and image shape;
* ``state/{left_ee_joint_states,right_ee_joint_states}`` with measured finger
  aperture, while every action dataset remains unchanged.

By default corrected HDF5 files are written below ``data_replayed`` and the
source dataset is untouched.  ``--standalone`` creates complete HDF5 files
from the lightweight replay inputs and does not require the old ``data``
directories.  Pass ``--in-place`` only when replacement of ``data_old`` is
intentional.  Long runs can be split across GPUs/processes with
``--num-shards`` and ``--shard-index``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rerender_head_camera import (  # noqa: E402
    ReplayValidationError,
    _assert_close,
    _dataset_options,
    _embodiment_directory_name,
    _json_value,
    build_task_args,
    create_task,
    encode_jpeg_frames,
    episode_indices,
    install_local_gripper_planner,
    load_saved_inputs,
    shutdown_task_planners,
    validate_scene_info,
)


DEFAULT_CONFIGS = ("piper_x_demo_clean", "piper_x_demo_randomized")
CAMERA_MAP = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
ARM_FIELDS = (
    "left_arm_joint_states",
    "right_arm_joint_states",
)
POSE_FIELDS = (
    "left_ee_poses",
    "right_ee_poses",
)
GRIPPER_FIELDS = ("left_ee_joint_states", "right_ee_joint_states")
REPLAY_VERSION = "actual-gripper-three-camera-v2"
MARKER_VERSION = "robotwin_replay_version"
MARKER_FINGERPRINT = "robotwin_replay_camera_fingerprint"
TRAJECTORY_EPISODE_RE = re.compile(r"episode(\d+)\.pkl$")


@dataclass(frozen=True)
class EpisodeJob:
    config: str
    task: str
    episode_dir: Path
    index: int

    @property
    def source_hdf5(self) -> Path:
        return self.episode_dir / "data" / f"episode_{self.index:07d}.hdf5"

    @property
    def label(self) -> str:
        return f"{self.config}/{self.task}/episode_{self.index:07d}"


def simulation_tasks(data_root: Path) -> tuple[str, ...]:
    tasks_path = data_root / "tasks.json"
    with tasks_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    tasks = value.get("simulation_tasks") if isinstance(value, dict) else None
    if not isinstance(tasks, list) or len(tasks) != 20 or len(set(tasks)) != 20:
        raise ValueError(f"{tasks_path} must contain exactly 20 unique simulation_tasks")
    return tuple(str(task) for task in tasks)


def iter_jobs(
    data_root: Path,
    configs: Sequence[str],
    selected_tasks: set[str] | None,
    start: int | None,
    stop: int | None,
    selected_episodes: set[int] | None,
    num_shards: int,
    shard_index: int,
    standalone: bool = False,
) -> list[EpisodeJob]:
    expected_tasks = simulation_tasks(data_root)
    if selected_tasks is not None:
        unknown = selected_tasks - set(expected_tasks)
        if unknown:
            raise ValueError(f"Tasks are not in simulation_tasks: {sorted(unknown)}")

    jobs: list[EpisodeJob] = []
    ordinal = 0
    for config in configs:
        config_path = REPO_ROOT / "env_cfg" / "task_config" / f"{config}.yml"
        with config_path.open("r", encoding="utf-8") as stream:
            import yaml

            config_args = yaml.safe_load(stream)
        embodiment = _embodiment_directory_name(config_args["embodiment"])
        config_dir = data_root / config
        if not config_dir.is_dir():
            raise FileNotFoundError(f"Dataset configuration directory not found: {config_dir}")

        for task in expected_tasks:
            if selected_tasks is not None and task not in selected_tasks:
                continue
            episode_dir = config_dir / task / embodiment
            if not episode_dir.is_dir():
                raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
            indices = (
                standalone_episode_indices(episode_dir)
                if standalone
                else episode_indices(episode_dir / "data")
            )
            if not indices:
                source = episode_dir / "_traj_data" if standalone else episode_dir / "data"
                kind = "trajectory" if standalone else "HDF5"
                raise FileNotFoundError(f"No {kind} episodes found under {source}")
            for index in indices:
                selected = (
                    (start is None or index >= start)
                    and (stop is None or index < stop)
                    and (selected_episodes is None or index in selected_episodes)
                )
                if not selected:
                    continue
                if ordinal % num_shards == shard_index:
                    jobs.append(EpisodeJob(config, task, episode_dir, index))
                ordinal += 1
    return jobs


def standalone_episode_indices(episode_dir: Path) -> list[int]:
    """Validate and enumerate a lightweight episode package without HDF5."""
    seed_path = episode_dir / "seed.txt"
    seeds = seed_path.read_text(encoding="utf-8").split()
    expected = set(range(len(seeds)))
    trajectory_dir = episode_dir / "_traj_data"
    trajectory_indices = {
        int(match.group(1))
        for path in trajectory_dir.iterdir()
        if path.is_file() and (match := TRAJECTORY_EPISODE_RE.fullmatch(path.name))
    }
    if trajectory_indices != expected:
        missing = sorted(expected - trajectory_indices)
        extra = sorted(trajectory_indices - expected)
        raise ReplayValidationError(
            f"{episode_dir}: trajectory/seed mismatch; missing={missing}, extra={extra}"
        )

    scene_path = episode_dir / "scene_info.json"
    with scene_path.open("r", encoding="utf-8") as stream:
        scene_info = json.load(stream)
    missing_scenes = [index for index in expected if f"episode_{index}" not in scene_info]
    missing_instructions = [
        index
        for index in expected
        if not (episode_dir / "instruction" / f"episode_{index:07d}.json").is_file()
    ]
    if missing_scenes or missing_instructions:
        raise ReplayValidationError(
            f"{episode_dir}: incomplete standalone inputs; "
            f"missing scenes={missing_scenes}, missing instructions={missing_instructions}"
        )
    return sorted(expected)


def replay_fingerprint(args: Mapping[str, Any], *, standalone: bool = False) -> str:
    """Hash every current setting that can alter the three rendered cameras."""
    digest = hashlib.sha256()
    payload = {
        "version": REPLAY_VERSION,
        "standalone": standalone,
        "camera_shader": args.get("camera_shader"),
        "camera": args.get("camera"),
        "domain_randomization": args.get("domain_randomization"),
        "left": {
            "static_camera_list": args["left_embodiment_config"].get("static_camera_list"),
            "head_camera_calibration": args["left_embodiment_config"].get(
                "head_camera_calibration"
            ),
            "wrist_camera_calibration": args["left_embodiment_config"].get(
                "wrist_camera_calibration"
            ),
        },
        "right": {
            "wrist_camera_calibration": args["right_embodiment_config"].get(
                "wrist_camera_calibration"
            ),
        },
    }
    digest.update(
        json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":")).encode()
    )
    for side in ("left", "right"):
        config = args[f"{side}_embodiment_config"]
        urdf = (Path(args[f"{side}_robot_file"]) / config["urdf_path"]).resolve()
        digest.update(str(urdf).encode())
        digest.update(urdf.read_bytes())
    return digest.hexdigest()


def destination_hdf5(job: EpisodeJob, data_root: Path, output_root: Path) -> Path:
    return output_root / job.source_hdf5.relative_to(data_root)


def episode_is_complete(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as episode:
            if episode.attrs.get(MARKER_VERSION) != REPLAY_VERSION:
                return False
            if episode.attrs.get(MARKER_FINGERPRINT) != fingerprint:
                return False
            lengths = []
            for camera in CAMERA_MAP.values():
                for field in ("colors", "intrinsic_matrix", "extrinsics_matrix"):
                    lengths.append(len(episode[f"vision/{camera}/{field}"]))
            for field in GRIPPER_FIELDS:
                lengths.append(len(episode[f"state/{field}"]))
            return bool(lengths) and min(lengths) > 0 and len(set(lengths)) == 1
    except (KeyError, OSError, TypeError):
        return False


def empty_capture() -> dict[str, Any]:
    return {
        "cameras": {
            source: {"frames": [], "intrinsics": [], "extrinsics": []}
            for source in CAMERA_MAP
        },
        "commands": {
            field: [] for field in ARM_FIELDS + POSE_FIELDS + GRIPPER_FIELDS
        },
        "actual_grippers": {field: [] for field in GRIPPER_FIELDS},
    }


def install_three_camera_capture(task: Any) -> dict[str, Any]:
    """Capture calibrated cameras and physical finger state at collection timing."""
    captured = empty_capture()

    def capture(instance: Any) -> None:
        # Match Base_Task._take_picture: update poses/render once, then trigger
        # all camera sensors.  This preserves frame/action alignment and RNG use.
        instance._update_render()
        instance.cameras.update_picture()
        rgb = instance.cameras.get_rgb()
        camera_config = instance.cameras.get_config()

        for source_name in CAMERA_MAP:
            if source_name not in rgb or source_name not in camera_config:
                raise ReplayValidationError(f"Replay is missing camera {source_name}")
            config = camera_config[source_name]
            extrinsic = config.get("cam2world_gl")
            if extrinsic is None:
                extrinsic = config.get("extrinsic_cv")
            if extrinsic is None:
                raise ReplayValidationError(f"Camera {source_name} has no extrinsics")
            camera = captured["cameras"][source_name]
            camera["frames"].append(np.asarray(rgb[source_name]["rgb"]).copy())
            camera["intrinsics"].append(np.asarray(config["intrinsic_cv"]).copy())
            camera["extrinsics"].append(np.asarray(extrinsic).copy())

        left = instance.robot.get_left_arm_jointState()
        right = instance.robot.get_right_arm_jointState()
        commands = captured["commands"]
        commands["left_arm_joint_states"].append(np.asarray(left[:-1]))
        commands["left_ee_joint_states"].append(np.asarray([left[-1]]))
        commands["right_arm_joint_states"].append(np.asarray(right[:-1]))
        commands["right_ee_joint_states"].append(np.asarray([right[-1]]))
        commands["left_ee_poses"].append(np.asarray(instance.get_arm_pose("left")))
        commands["right_ee_poses"].append(np.asarray(instance.get_arm_pose("right")))

        actual = captured["actual_grippers"]
        actual["left_ee_joint_states"].append(
            np.asarray([instance.robot.get_left_gripper_actual_val()], dtype=np.float32)
        )
        actual["right_ee_joint_states"].append(
            np.asarray([instance.robot.get_right_gripper_actual_val()], dtype=np.float32)
        )
        instance.FRAME_IDX += 1

    task._take_picture = MethodType(capture, task)
    return captured


def validate_capture(captured: Mapping[str, Any], label: str) -> int:
    """Validate a self-contained capture and return its output frame count."""
    lengths: dict[str, int] = {}
    for source_name, camera in captured["cameras"].items():
        for field in ("frames", "intrinsics", "extrinsics"):
            lengths[f"{source_name}/{field}"] = len(camera[field])
    for field, values in captured["commands"].items():
        lengths[f"commands/{field}"] = len(values)
    for field, values in captured["actual_grippers"].items():
        lengths[f"actual_grippers/{field}"] = len(values)

    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1 or not unique_lengths or next(iter(unique_lengths)) < 2:
        raise ReplayValidationError(f"{label}: inconsistent replay lengths: {lengths}")
    capture_count = next(iter(unique_lengths))
    for field, values in captured["actual_grippers"].items():
        array = np.asarray(values)
        if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
            raise ReplayValidationError(f"{label}: {field} contains invalid values")
    return capture_count - 1


def validate_replay(
    source_hdf5: Path,
    captured: Mapping[str, Any],
    state_atol: float,
    pose_atol: float = 2e-3,
) -> int:
    """Check replay alignment without comparing obsolete gripper state values."""
    captured_frame_count = validate_capture(captured, str(source_hdf5))
    with h5py.File(source_hdf5, "r") as source:
        frame_counts = []
        for target_name in CAMERA_MAP.values():
            try:
                frame_counts.append(len(source[f"vision/{target_name}/colors"]))
            except KeyError as exc:
                raise ReplayValidationError(
                    f"{source_hdf5}: missing vision/{target_name}/colors"
                ) from exc
        if len(set(frame_counts)) != 1:
            raise ReplayValidationError(
                f"{source_hdf5}: source camera frame counts differ: {frame_counts}"
            )
        frame_count = frame_counts[0]
        if captured_frame_count != frame_count:
            raise ReplayValidationError(
                f"{source_hdf5}: replay has {captured_frame_count} frames, "
                f"expected {frame_count}"
            )

        commands = captured["commands"]
        for field in ARM_FIELDS + POSE_FIELDS:
            values = np.asarray(commands[field])
            atol = pose_atol if field in POSE_FIELDS else state_atol
            _assert_close(
                f"state/{field}", values[:-1], np.asarray(source[f"state/{field}"]), atol
            )
            _assert_close(
                f"action/{field}",
                values[1:],
                np.asarray(source[f"action/{field}"]),
                atol,
            )
        # Old gripper state contains command targets and will be replaced by
        # measured aperture.  The action side must still match exactly.
        for field in GRIPPER_FIELDS:
            values = np.asarray(commands[field])
            _assert_close(
                f"action/{field}",
                values[1:],
                np.asarray(source[f"action/{field}"]),
                state_atol,
            )

    return frame_count


def load_instructions(episode_dir: Path, index: int) -> list[str]:
    path = episode_dir / "instruction" / f"episode_{index:07d}.json"
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    instructions = value.get("seen") if isinstance(value, dict) else None
    if not isinstance(instructions, list):
        raise ReplayValidationError(f"{path}: expected a 'seen' instruction list")
    result = [str(item) for item in instructions if str(item)]
    return result or [""]


def _replace_dataset(group: h5py.Group, name: str, data: Any, **kwargs: Any) -> None:
    old = group[name]
    attrs = dict(old.attrs)
    options = _dataset_options(old)
    del group[name]
    dataset = group.create_dataset(name, data=data, **options, **kwargs)
    for key, value in attrs.items():
        dataset.attrs[key] = value


def atomic_write_episode(
    source_hdf5: Path,
    destination: Path,
    captured: Mapping[str, Any],
    frame_count: int,
    fingerprint: str,
) -> None:
    """Copy, update, fsync and atomically publish one corrected episode."""
    encoded_cameras: dict[str, tuple[list[bytes], int]] = {}
    for source_name, camera in captured["cameras"].items():
        encoded_cameras[source_name] = encode_jpeg_frames(camera["frames"][:frame_count])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.replay.tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        import shutil

        shutil.copy2(source_hdf5, temp_path)
        with h5py.File(temp_path, "r+") as target:
            for source_name, target_name in CAMERA_MAP.items():
                camera = captured["cameras"][source_name]
                frames = camera["frames"][:frame_count]
                encoded, max_len = encoded_cameras[source_name]
                group = target[f"vision/{target_name}"]
                if len(group["colors"]) != frame_count:
                    raise ReplayValidationError(
                        f"{source_hdf5}: {target_name} frame count changed"
                    )
                _replace_dataset(group, "colors", encoded, dtype=f"S{max_len}")
                _replace_dataset(
                    group,
                    "intrinsic_matrix",
                    np.asarray(camera["intrinsics"][:frame_count], dtype=np.float32),
                )
                _replace_dataset(
                    group,
                    "extrinsics_matrix",
                    np.asarray(camera["extrinsics"][:frame_count], dtype=np.float32),
                )
                _replace_dataset(
                    group,
                    "shape",
                    np.asarray(frames[0].shape, dtype=np.int32),
                )

            for field in GRIPPER_FIELDS:
                values = np.asarray(
                    captured["actual_grippers"][field][:frame_count], dtype=np.float32
                )
                _replace_dataset(target["state"], field, values)

            target.attrs[MARKER_VERSION] = REPLAY_VERSION
            target.attrs[MARKER_FINGERPRINT] = fingerprint
            target.attrs["robotwin_replay_source"] = str(source_hdf5)
            target.flush()

        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_create_episode(
    destination: Path,
    captured: Mapping[str, Any],
    frame_count: int,
    fingerprint: str,
    instructions: Sequence[str],
    frequency: int,
    source_label: str,
) -> None:
    """Create and atomically publish a complete episode without a source HDF5."""
    encoded_cameras: dict[str, tuple[list[bytes], int]] = {}
    for source_name, camera in captured["cameras"].items():
        encoded_cameras[source_name] = encode_jpeg_frames(camera["frames"][:frame_count])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.replay.tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        with h5py.File(temp_path, "w") as target:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            target.attrs["source_format"] = "RoboTwin"
            target.attrs["source_path"] = "standalone_replay"
            target.attrs[MARKER_VERSION] = REPLAY_VERSION
            target.attrs[MARKER_FINGERPRINT] = fingerprint
            target.attrs["robotwin_replay_source"] = source_label
            target.create_dataset("data_format_version", data="v1.0", dtype=string_dtype)
            target.create_dataset(
                "instructions",
                data=json.dumps(list(instructions), ensure_ascii=False),
                dtype=string_dtype,
            )
            target.create_group("additional_info").create_dataset(
                "frequency", data=np.asarray(frequency, dtype=np.int32)
            )

            state = target.create_group("state")
            action = target.create_group("action")
            for field in ARM_FIELDS + POSE_FIELDS:
                values = np.asarray(captured["commands"][field], dtype=np.float32)
                state.create_dataset(field, data=values[:frame_count])
                action.create_dataset(field, data=values[1 : frame_count + 1])
            for field in GRIPPER_FIELDS:
                actual = np.asarray(
                    captured["actual_grippers"][field], dtype=np.float32
                )
                commands = np.asarray(captured["commands"][field], dtype=np.float32)
                state.create_dataset(field, data=actual[:frame_count])
                action.create_dataset(field, data=commands[1 : frame_count + 1])

            vision = target.create_group("vision")
            for source_name, target_name in CAMERA_MAP.items():
                camera = captured["cameras"][source_name]
                frames = camera["frames"][:frame_count]
                encoded, max_len = encoded_cameras[source_name]
                group = vision.create_group(target_name)
                group.create_dataset("colors", data=encoded, dtype=f"S{max_len}")
                group.create_dataset(
                    "intrinsic_matrix",
                    data=np.asarray(camera["intrinsics"][:frame_count], dtype=np.float32),
                )
                group.create_dataset(
                    "extrinsics_matrix",
                    data=np.asarray(camera["extrinsics"][:frame_count], dtype=np.float32),
                )
                group.create_dataset(
                    "shape", data=np.asarray(frames[0].shape, dtype=np.int32)
                )
            target.flush()

        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def replay_episode(
    task: Any,
    job: EpisodeJob,
    destination: Path,
    fingerprint: str,
    *,
    state_atol: float,
    pose_atol: float,
    clear_cache: bool,
    camera_shader: str | None = None,
    standalone: bool = False,
) -> None:
    seed, trajectory, expected_scene = load_saved_inputs(job.episode_dir, job.index)
    args = build_task_args(job.task, job.config, job.episode_dir)
    if camera_shader is not None:
        args["camera_shader"] = camera_shader
    initialized = False
    try:
        task.setup_demo(now_ep_num=job.index, seed=seed, **args)
        initialized = True
        task.set_path_lst(
            {
                "need_plan": False,
                "left_joint_path": trajectory["left_joint_path"],
                "right_joint_path": trajectory["right_joint_path"],
            }
        )
        captured = install_three_camera_capture(task)
        actual_scene = task.play_once()
        frame_count = (
            validate_capture(captured, job.label)
            if standalone
            else validate_replay(job.source_hdf5, captured, state_atol, pose_atol)
        )
        validate_scene_info(expected_scene, actual_scene, job.label)
        if not task.plan_success or not task.check_success():
            raise ReplayValidationError(f"{job.label}: replay did not finish successfully")
        if standalone:
            atomic_create_episode(
                destination,
                captured,
                frame_count,
                fingerprint,
                load_instructions(job.episode_dir, job.index),
                int(args.get("save_freq") or 15),
                job.label,
            )
        else:
            atomic_write_episode(
                job.source_hdf5, destination, captured, frame_count, fingerprint
            )
    finally:
        if initialized:
            task.close_env(clear_cache=clear_cache)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_old")
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--output-root",
        type=Path,
        help="Corrected dataset root (default: <repo>/data_replayed)",
    )
    output.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace source HDF5 files under --data-root",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help=(
            "Create complete HDF5 files from seed/trajectory/scene/instruction inputs; "
            "do not read old HDF5 files"
        ),
    )
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--tasks", nargs="+", help="Subset of the 20 simulation tasks")
    parser.add_argument("--episode", type=int, action="append", dest="episodes")
    parser.add_argument("--start", type=int, help="Inclusive episode index per task")
    parser.add_argument("--stop", type=int, help="Exclusive episode index per task")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--state-atol", type=float, default=1e-4)
    parser.add_argument("--pose-atol", type=float, default=2e-3)
    parser.add_argument("--clear-cache-every", type=int, default=5)
    parser.add_argument(
        "--camera-shader",
        help="Override camera_shader from the collection config (useful for smoke tests)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_args(argv)
    data_root = cli.data_root.resolve()
    output_root = (
        data_root
        if cli.in_place
        else (cli.output_root or (REPO_ROOT / "data_replayed")).resolve()
    )
    if output_root == data_root and not cli.in_place:
        raise SystemExit("Use --in-place explicitly to replace files under --data-root")
    if cli.standalone and cli.in_place:
        raise SystemExit("--standalone cannot be combined with --in-place")
    if cli.num_shards <= 0 or not 0 <= cli.shard_index < cli.num_shards:
        raise SystemExit("--num-shards must be positive and 0 <= --shard-index < --num-shards")
    if cli.start is not None and cli.stop is not None and cli.start >= cli.stop:
        raise SystemExit("--start must be smaller than --stop")
    if cli.state_atol < 0 or cli.pose_atol < 0 or cli.clear_cache_every <= 0:
        raise SystemExit(
            "--state-atol/--pose-atol must be nonnegative and "
            "--clear-cache-every positive"
        )

    jobs = iter_jobs(
        data_root,
        cli.configs,
        set(cli.tasks) if cli.tasks else None,
        cli.start,
        cli.stop,
        set(cli.episodes) if cli.episodes else None,
        cli.num_shards,
        cli.shard_index,
        cli.standalone,
    )
    fingerprints: dict[tuple[str, str], str] = {}
    pending: list[tuple[EpisodeJob, Path, str]] = []
    for job in jobs:
        fingerprint_key = (job.config, job.task)
        if fingerprint_key not in fingerprints:
            args = build_task_args(job.task, job.config, job.episode_dir)
            if cli.camera_shader is not None:
                args["camera_shader"] = cli.camera_shader
            fingerprints[fingerprint_key] = replay_fingerprint(
                args, standalone=cli.standalone
            )
        fingerprint = fingerprints[fingerprint_key]
        destination = destination_hdf5(job, data_root, output_root)
        if not cli.force and episode_is_complete(destination, fingerprint):
            status = "SKIP"
        else:
            pending.append((job, destination, fingerprint))
            status = "TODO"
        if cli.dry_run:
            print(f"{status} {job.label} -> {destination}")

    print(
        f"Selected {len(jobs)} episode(s): {len(jobs) - len(pending)} complete, "
        f"{len(pending)} pending; output={output_root}"
    )
    if cli.dry_run or not pending:
        return 0

    import multiprocessing as mp

    from sapien.render import clear_cache
    from test_render import Sapien_TEST

    mp.set_start_method("spawn", force=True)
    install_local_gripper_planner()
    try:
        Sapien_TEST()
    except SystemExit as exc:
        raise RuntimeError("SAPIEN renderer initialization failed") from exc

    failures: list[tuple[str, str]] = []
    active_key: tuple[str, str] | None = None
    task = None
    try:
        for position, (job, destination, fingerprint) in enumerate(pending, start=1):
            key = (job.config, job.task)
            if key != active_key:
                if task is not None:
                    shutdown_task_planners(task)
                task = create_task(job.task)
                active_key = key
            print(f"[{position}/{len(pending)}] REPLAY {job.label}", flush=True)
            try:
                replay_episode(
                    task,
                    job,
                    destination,
                    fingerprint,
                    state_atol=cli.state_atol,
                    pose_atol=cli.pose_atol,
                    clear_cache=position % cli.clear_cache_every == 0,
                    camera_shader=cli.camera_shader,
                    standalone=cli.standalone,
                )
            except Exception as exc:
                failures.append((job.label, str(exc)))
                print(f"[{position}/{len(pending)}] FAIL   {job.label}: {exc}", flush=True)
                if not cli.keep_going:
                    raise
            else:
                print(f"[{position}/{len(pending)}] DONE   {job.label}", flush=True)
    finally:
        if task is not None:
            shutdown_task_planners(task)
        clear_cache()

    if failures:
        print(f"Replay completed with {len(failures)} failure(s):")
        for label, error in failures:
            print(f"  {label}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
