#!/usr/bin/env python3
"""Convert raw ZInD tours into the HPR-Loc dataset layout."""

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def rotate_points(points, angle_degrees):
    points = np.asarray(points, dtype=np.float64)
    angle = math.radians(angle_degrees)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    return points @ rotation.T


def transform_points(points, pano_transform, floor_scale, global_rotation):
    transformed = rotate_points(points, pano_transform["rotation"])
    transformed *= float(pano_transform["scale"])
    transformed += np.asarray(pano_transform["translation"], dtype=np.float64)
    transformed *= float(floor_scale)
    return rotate_points(transformed, global_rotation)


def iter_floor_panos(data, floor_name):
    floor = data.get("merger", {}).get(floor_name, {})
    for complete_room_name in sorted(floor):
        partial_rooms = floor[complete_room_name]
        for partial_room_name in sorted(partial_rooms):
            panos = partial_rooms[partial_room_name]
            for pano_name in sorted(panos):
                yield complete_room_name, partial_room_name, pano_name, panos[pano_name]


def first_pano(partial_rooms):
    first_partial_name = sorted(partial_rooms)[0]
    panos = partial_rooms[first_partial_name]
    return panos[sorted(panos)[0]]


def segment_pairs(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return []
    if len(values) % 3 == 0:
        return [values[index : index + 2] for index in range(0, len(values), 3)]
    if len(values) % 2 == 0:
        return [values[index : index + 2] for index in range(0, len(values), 2)]
    return []


def extract_floor_geometry(data, floor_name):
    floor_scale = data["scale_meters_per_coordinate"].get(floor_name)
    if floor_scale is None:
        return None

    redraw_transform = data.get("floorplan_to_redraw_transformation", {}).get(
        floor_name, {}
    )
    global_rotation = -float(redraw_transform.get("rotation", 0.0))
    rooms = []
    doors = []
    openings = []
    windows = []
    floor = data.get("merger", {}).get(floor_name, {})

    for partial_rooms in floor.values():
        pano = first_pano(partial_rooms)
        transform = pano["floor_plan_transformation"]
        layout = pano.get("layout_complete", {})
        vertices = layout.get("vertices", [])
        if len(vertices) >= 3:
            rooms.append(
                transform_points(
                    vertices,
                    transform,
                    floor_scale,
                    global_rotation,
                )
            )

        for source, destination in (
            ("doors", doors),
            ("openings", openings),
            ("windows", windows),
        ):
            for pair in segment_pairs(layout.get(source, [])):
                destination.append(
                    transform_points(
                        pair,
                        transform,
                        floor_scale,
                        global_rotation,
                    )
                )

    cameras = []
    for _, _, pano_name, pano in iter_floor_panos(data, floor_name):
        transform = pano["floor_plan_transformation"]
        position = np.asarray(transform["translation"], dtype=np.float64) * floor_scale
        position = rotate_points(position[None], global_rotation)[0]
        pano_rotation = float(transform["rotation"]) + global_rotation
        cameras.append(
            {
                "pano_name": pano_name,
                "image_path": pano["image_path"],
                "position_m": position,
                "theta": math.radians(90.0 - pano_rotation) % (2.0 * math.pi),
                "is_inside": bool(pano.get("is_inside", True)),
                "is_primary": bool(pano.get("is_primary", False)),
            }
        )

    return {
        "rooms": rooms,
        "doors": doors,
        "openings": openings,
        "windows": windows,
        "cameras": cameras,
    }


def shift_geometry(geometry, padding_m):
    point_sets = geometry["rooms"] + [
        camera["position_m"][None] for camera in geometry["cameras"]
    ]
    all_points = np.concatenate(point_sets, axis=0)
    minimum = all_points.min(axis=0) - padding_m
    maximum = all_points.max(axis=0) + padding_m

    shifted = {}
    for key in ("rooms", "doors", "openings", "windows"):
        shifted[key] = [points - minimum for points in geometry[key]]
    shifted["cameras"] = []
    for camera in geometry["cameras"]:
        camera = dict(camera)
        camera["position_m"] = camera["position_m"] - minimum
        shifted["cameras"].append(camera)
    shifted["size_m"] = maximum - minimum
    shifted["origin_shift_m"] = -minimum
    return shifted


def points_to_cv(points, pixels_per_meter):
    return np.rint(points * pixels_per_meter).astype(np.int32).reshape(-1, 1, 2)


def render_map(geometry, pixels_per_meter, wall_thickness_m):
    width = max(1, int(math.ceil(geometry["size_m"][0] * pixels_per_meter)) + 1)
    height = max(1, int(math.ceil(geometry["size_m"][1] * pixels_per_meter)) + 1)
    image = np.full((height, width), 255, dtype=np.uint8)
    wall_thickness = max(2, int(round(wall_thickness_m * pixels_per_meter)))
    gap_thickness = wall_thickness + max(2, int(round(0.04 * pixels_per_meter)))

    for room in geometry["rooms"]:
        cv2.polylines(
            image,
            [points_to_cv(room, pixels_per_meter)],
            isClosed=True,
            color=0,
            thickness=wall_thickness,
            lineType=cv2.LINE_8,
        )
    for window in geometry["windows"]:
        cv2.line(
            image,
            tuple(points_to_cv(window, pixels_per_meter).reshape(-1, 2)[0]),
            tuple(points_to_cv(window, pixels_per_meter).reshape(-1, 2)[1]),
            0,
            wall_thickness,
            cv2.LINE_8,
        )
    for gap in geometry["doors"] + geometry["openings"]:
        points = points_to_cv(gap, pixels_per_meter).reshape(-1, 2)
        cv2.line(
            image,
            tuple(points[0]),
            tuple(points[1]),
            255,
            gap_thickness,
            cv2.LINE_8,
        )
    return image


def pano_to_perspective(image, fov_degrees, yaw_degrees, output_size):
    equ_height, equ_width = image.shape[:2]
    output_height, output_width = output_size
    horizontal_fov = float(fov_degrees)
    vertical_fov = output_height / output_width * horizontal_fov
    radius = 128.0

    horizontal_angle = math.radians((180.0 - horizontal_fov) / 2.0)
    horizontal_length = (
        2.0 * radius * math.sin(math.radians(horizontal_fov / 2.0))
        / math.sin(horizontal_angle)
    )
    vertical_angle = math.radians((180.0 - vertical_fov) / 2.0)
    vertical_length = (
        2.0 * radius * math.sin(math.radians(vertical_fov / 2.0))
        / math.sin(vertical_angle)
    )

    x_map = np.full((output_height, output_width), radius, dtype=np.float32)
    y_map = np.tile(
        (np.arange(output_width) - (output_width - 1) / 2.0)
        * horizontal_length
        / (output_width - 1),
        (output_height, 1),
    )
    z_map = -np.tile(
        (np.arange(output_height) - (output_height - 1) / 2.0)
        * vertical_length
        / (output_height - 1),
        (output_width, 1),
    ).T
    distance = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.stack(
        [
            radius * x_map / distance,
            radius * y_map / distance,
            radius * z_map / distance,
        ],
        axis=-1,
    )

    angle = math.radians(yaw_degrees - 180.0)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    xyz = xyz.reshape(-1, 3) @ rotation.T
    latitude = np.arcsin(np.clip(xyz[:, 2] / radius, -1.0, 1.0))
    longitude = np.arctan2(xyz[:, 1], xyz[:, 0])
    map_x = ((longitude / math.pi + 1.0) * (equ_width - 1) / 2.0).reshape(
        output_height, output_width
    )
    map_y = ((-latitude / math.pi * 2.0 + 1.0) * (equ_height - 1) / 2.0).reshape(
        output_height, output_width
    )
    return cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )


def ray_offsets(num_rays, fov_degrees):
    focal_length = (num_rays / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    pixel_coordinates = np.arange(num_rays) - (num_rays - 1) / 2.0
    return np.flip(np.arctan2(pixel_coordinates, focal_length))


def raycast_depth40(
    floor_map,
    position_px,
    theta,
    offsets,
    pixels_per_meter,
    max_distance_m,
):
    step_m = 1.0 / pixels_per_meter
    distances = np.arange(0.05, max_distance_m + step_m, step_m)
    angles = theta + offsets
    sample_x = np.rint(
        position_px[0] + np.cos(angles)[:, None] * distances[None] * pixels_per_meter
    ).astype(np.int32)
    sample_y = np.rint(
        position_px[1] + np.sin(angles)[:, None] * distances[None] * pixels_per_meter
    ).astype(np.int32)
    valid = (
        (sample_x >= 0)
        & (sample_x < floor_map.shape[1])
        & (sample_y >= 0)
        & (sample_y < floor_map.shape[0])
    )
    hits = np.ones_like(valid, dtype=bool)
    hits[valid] = floor_map[sample_y[valid], sample_x[valid]] < 128
    first_hit = hits.argmax(axis=1)
    radial_depth = distances[first_hit]
    no_hit = ~hits.any(axis=1)
    radial_depth[no_hit] = max_distance_m
    return np.minimum(radial_depth * np.cos(offsets), max_distance_m)


def write_scene(
    tour_dir,
    floor_name,
    geometry,
    output_root,
    args,
):
    scene_name = f"scene_{int(tour_dir.name):04d}_{floor_name}"
    scene_dir = output_root / scene_name
    if scene_dir.exists() and args.overwrite:
        shutil.rmtree(scene_dir)
    elif scene_dir.exists():
        raise FileExistsError(
            f"{scene_dir} already exists; pass --overwrite to replace it"
        )
    rgb_dir = scene_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    floor_map = render_map(
        geometry,
        args.pixels_per_meter,
        args.wall_thickness_m,
    )
    cv2.imwrite(str(scene_dir / "map.png"), floor_map)
    offsets = ray_offsets(args.num_rays, args.fov)
    poses = []
    depths = []
    metadata = []
    sample_index = 0

    for pano_index, camera in enumerate(geometry["cameras"]):
        pano_path = tour_dir / camera["image_path"]
        panorama = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
        if panorama is None:
            print(f"warning: missing panorama {pano_path}")
            continue

        position_px = camera["position_m"] * args.pixels_per_meter
        for view_index, yaw in enumerate(args.yaws):
            theta = (camera["theta"] - math.radians(yaw)) % (2.0 * math.pi)
            perspective = pano_to_perspective(
                panorama,
                args.fov,
                yaw,
                (args.image_height, args.image_width),
            )
            image_name = f"{pano_index:05d}-{view_index}.jpg"
            cv2.imwrite(
                str(rgb_dir / image_name),
                perspective,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            depth = raycast_depth40(
                floor_map,
                position_px,
                theta,
                offsets,
                args.pixels_per_meter,
                args.max_depth_m,
            )
            poses.append((position_px[0], position_px[1], theta))
            depths.append(depth)
            metadata.append(
                {
                    "sample_index": sample_index,
                    "image": f"rgb/{image_name}",
                    "source_panorama": camera["image_path"],
                    "pano_name": camera["pano_name"],
                    "yaw_degrees": yaw,
                    "is_inside": camera["is_inside"],
                    "is_primary": camera["is_primary"],
                }
            )
            sample_index += 1

    with (scene_dir / "poses_map.txt").open("w", encoding="utf-8") as handle:
        for pose in poses:
            handle.write(f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.9f}\n")
    with (scene_dir / "depth40.txt").open("w", encoding="utf-8") as handle:
        for depth in depths:
            handle.write(" ".join(f"{value:.6f}" for value in depth) + "\n")
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "tour_id": tour_dir.name,
                "floor_name": floor_name,
                "map_res": 1.0 / args.pixels_per_meter,
                "fov_degrees": args.fov,
                "origin_shift_m": geometry["origin_shift_m"].tolist(),
                "samples": metadata,
            },
            handle,
            indent=2,
        )
    return scene_name, len(poses)


def normalized_tour_id(value):
    try:
        return f"{int(value):04d}"
    except ValueError:
        return value


def load_partition(path):
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        partition = json.load(handle)
    return {
        split: {normalized_tour_id(tour_id) for tour_id in tour_ids}
        for split, tour_ids in partition.items()
    }


def find_split(tour_id, partition):
    if partition is None:
        return "train"
    normalized = normalized_tour_id(tour_id)
    for split in ("train", "val", "test"):
        if normalized in partition.get(split, set()):
            return split
    return None


def load_semrayloc_splits(processed_root):
    split_path = processed_root / "split.yaml"
    with split_path.open("r", encoding="utf-8") as handle:
        split_data = yaml.safe_load(handle)

    scene_splits = {}
    for split in ("train", "val", "test"):
        for scene_name in split_data.get(split, []):
            if scene_name in scene_splits:
                raise ValueError(f"scene appears in multiple splits: {scene_name}")
            scene_splits[scene_name] = split
    return scene_splits


def write_semrayloc_scene(raw_root, processed_scene_dir, output_root, args):
    scene_name = processed_scene_dir.name
    scene_dir = output_root / scene_name
    completion_marker = scene_dir / ".complete"
    semantic_map_source = processed_scene_dir / "floorplan_semantic.png"

    def copy_semantic_map():
        if semantic_map_source.is_file():
            shutil.copy2(semantic_map_source, scene_dir / "floorplan_semantic.png")

    if scene_dir.exists() and args.overwrite:
        shutil.rmtree(scene_dir)
    elif scene_dir.exists() and args.skip_existing:
        if completion_marker.is_file():
            copy_semantic_map()
            return sum(
                1
                for line in (scene_dir / "poses_map.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
        shutil.rmtree(scene_dir)
    elif scene_dir.exists():
        raise FileExistsError(
            f"{scene_dir} already exists; pass --overwrite to replace it or "
            "--skip-existing to resume completed scenes"
        )

    floor_map = cv2.imread(
        str(processed_scene_dir / "floorplan_walls_only.png"), cv2.IMREAD_GRAYSCALE
    )
    if floor_map is None:
        raise FileNotFoundError(f"missing wall map for {scene_name}")

    poses = [
        list(map(float, line.split()))[:3]
        for line in (processed_scene_dir / "poses.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    with (processed_scene_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    pano_paths = metadata["original_path"]
    if len(poses) != len(pano_paths):
        raise ValueError(
            f"pose/metadata length mismatch in {scene_name}: "
            f"{len(poses)} != {len(pano_paths)}"
        )

    home_id = int(scene_name.split("_")[1])
    home_dir = raw_root / f"{home_id:04d}"
    rgb_dir = scene_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(scene_dir / "map.png"), floor_map)
    copy_semantic_map()

    offsets = ray_offsets(args.num_rays, args.fov)
    output_poses = []
    output_depths = []
    output_metadata = []
    for pano_index, (pose, relative_pano_path) in enumerate(zip(poses, pano_paths)):
        panorama = cv2.imread(str(home_dir / relative_pano_path), cv2.IMREAD_COLOR)
        if panorama is None:
            print(f"warning: missing panorama {home_dir / relative_pano_path}")
            continue

        position_px = np.asarray(pose[:2], dtype=np.float64) * args.pixels_per_meter
        if not (
            0 <= position_px[0] < floor_map.shape[1]
            and 0 <= position_px[1] < floor_map.shape[0]
        ):
            print(f"warning: out-of-bounds pose in {scene_name}: {position_px}")
            continue

        for view_index, yaw in enumerate(args.yaws):
            theta = (float(pose[2]) - math.radians(yaw)) % (2.0 * math.pi)
            perspective = pano_to_perspective(
                panorama,
                args.fov,
                yaw,
                (args.image_height, args.image_width),
            )
            image_name = f"{pano_index:05d}-{view_index}.jpg"
            cv2.imwrite(
                str(rgb_dir / image_name),
                perspective,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            output_poses.append((position_px[0], position_px[1], theta))
            output_depths.append(
                raycast_depth40(
                    floor_map,
                    position_px,
                    theta,
                    offsets,
                    args.pixels_per_meter,
                    args.max_depth_m,
                )
            )
            output_metadata.append(
                {
                    "image": image_name,
                    "source_pano": relative_pano_path,
                    "source_index": pano_index,
                    "yaw_degrees": yaw,
                }
            )

    if not output_poses:
        shutil.rmtree(scene_dir)
        return 0

    with (scene_dir / "poses_map.txt").open("w", encoding="utf-8") as handle:
        for pose in output_poses:
            handle.write("{:.6f} {:.6f} {:.8f}\n".format(*pose))
    with (scene_dir / "depth40.txt").open("w", encoding="utf-8") as handle:
        for depth in output_depths:
            handle.write(" ".join(f"{value:.6f}" for value in depth) + "\n")
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"source_scene": scene_name, "samples": output_metadata}, handle)
    completion_marker.touch()
    return len(output_poses)


def convert_semrayloc_layout(raw_root, processed_root, output_root, args):
    scene_splits = load_semrayloc_splits(processed_root)
    requested_scenes = set(args.scene) if args.scene else None
    split_scenes = {"train": [], "val": [], "test": []}
    total_samples = 0

    for scene_name, split in sorted(scene_splits.items()):
        if requested_scenes is not None and scene_name not in requested_scenes:
            continue
        scene_dir = processed_root / scene_name
        if not scene_dir.is_dir():
            print(f"warning: missing processed scene {scene_dir}")
            continue
        sample_count = write_semrayloc_scene(raw_root, scene_dir, output_root, args)
        if sample_count:
            split_scenes[split].append(scene_name)
            total_samples += sample_count
            print(f"{scene_name}: {sample_count} views")

    with (output_root / "split.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(split_scenes, handle, sort_keys=False)
    print(
        f"wrote {sum(map(len, split_scenes.values()))} scenes and "
        f"{total_samples} views to {output_root}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path)
    parser.add_argument("--tour", action="append", help="Only convert these tour IDs.")
    parser.add_argument(
        "--semrayloc-processed-root",
        type=Path,
        help="Use SemRayLoc's processed maps, poses, and split.yaml as the source.",
    )
    parser.add_argument(
        "--scene", action="append", help="Only convert these SemRayLoc scene names."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume a SemRayLoc conversion by preserving scenes with a .complete marker.",
    )
    parser.add_argument("--fov", type=float, default=80.0)
    parser.add_argument("--yaws", type=float, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--pixels-per-meter", type=float, default=100.0)
    parser.add_argument("--padding-m", type=float, default=0.5)
    parser.add_argument("--wall-thickness-m", type=float, default=0.08)
    parser.add_argument("--num-rays", type=int, default=40)
    parser.add_argument("--max-depth-m", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.semrayloc_processed_root:
        convert_semrayloc_layout(
            args.raw_root,
            args.semrayloc_processed_root,
            args.output_root,
            args,
        )
        return

    partition = load_partition(args.partition)
    requested_tours = (
        {normalized_tour_id(tour_id) for tour_id in args.tour}
        if args.tour
        else None
    )
    split_scenes = {"train": [], "val": [], "test": []}
    total_samples = 0

    tour_dirs = sorted(
        path
        for path in args.raw_root.iterdir()
        if path.is_dir() and (path / "zind_data.json").is_file()
    )
    for tour_dir in tour_dirs:
        tour_id = normalized_tour_id(tour_dir.name)
        if requested_tours is not None and tour_id not in requested_tours:
            continue
        split = find_split(tour_dir.name, partition)
        if split is None:
            print(f"warning: tour {tour_dir.name} is absent from the partition")
            continue

        with (tour_dir / "zind_data.json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for floor_name in sorted(data.get("merger", {})):
            geometry = extract_floor_geometry(data, floor_name)
            if geometry is None or not geometry["rooms"] or not geometry["cameras"]:
                print(f"warning: skipping {tour_dir.name}/{floor_name}")
                continue
            geometry = shift_geometry(geometry, args.padding_m)
            scene_name, sample_count = write_scene(
                tour_dir,
                floor_name,
                geometry,
                args.output_root,
                args,
            )
            split_scenes[split].append(scene_name)
            total_samples += sample_count
            print(f"{scene_name}: {sample_count} views")

    with (args.output_root / "split.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(split_scenes, handle, sort_keys=False)
    print(
        f"wrote {sum(map(len, split_scenes.values()))} scenes and "
        f"{total_samples} views to {args.output_root}"
    )


if __name__ == "__main__":
    main()
