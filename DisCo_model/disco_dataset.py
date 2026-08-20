import os
import warnings
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DisCo_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_splits_path: str,
        split: str,
        floorplan_img_size: Tuple[int, int],
        pose_aug_params: dict = None,
        dataset_cfg: dict = None,
    ):
        self.data_folder = data_folder
        self.data_splits_path = data_splits_path
        self.split = split
        self.floorplan_img_size = floorplan_img_size
        self.pose_aug_params = pose_aug_params if pose_aug_params else {"enable": False}
        self.dataset_cfg = dataset_cfg or {}
        self.dataset_type = self.dataset_cfg.get("dataset_type", "auto").lower()
        self.map_res = float(self.dataset_cfg.get("map_res", self._default_map_res()))
        self.hard_negative_mode = self._get_hard_negative_mode()
        self.floorplan_representation = self.dataset_cfg.get(
            "floorplan_representation",
            "rgb",
        )
        self.local_map_representation = self.dataset_cfg.get(
            "local_map_representation",
            "semantic_onehot"
            if self.floorplan_representation == "semantic_onehot"
            else "gray",
        )
        if self.local_map_representation not in ("gray", "semantic_onehot"):
            raise ValueError(
                "local_map_representation must be 'gray' or 'semantic_onehot'."
            )
        self.map_pose_rot_aug = self.dataset_cfg.get("map_pose_rot_aug", {})
        self.map_pose_rot_aug_enable = bool(self.map_pose_rot_aug.get("enable", False))
        self.map_pose_rot_aug_p = float(self.map_pose_rot_aug.get("p", 0.0))
        self.map_pose_rot_aug_angles = self._parse_rotation_angles(
            self.map_pose_rot_aug.get("angles", [0, 90, 180, 270])
        )

        with open(self.data_splits_path, "r", encoding="utf-8") as f:
            data_splits = yaml.safe_load(f)

        self._validate_splits(data_splits)
        self.data_split = ["".join(x.split()) for x in data_splits[self.split]]
        self.data = self._load_data(self.data_folder, self.data_split)
        if not self.data:
            raise RuntimeError(
                f"No samples were loaded for split '{self.split}' from "
                f"'{self.data_folder}'."
            )

    def _validate_splits(self, data_splits):
        if not isinstance(data_splits, dict):
            raise ValueError(
                f"Split file '{self.data_splits_path}' must contain a mapping."
            )
        if self.split not in data_splits:
            raise KeyError(
                f"Split '{self.split}' is not defined in '{self.data_splits_path}'."
            )

        normalized_splits = {}
        for split_name, scenes in data_splits.items():
            if not isinstance(scenes, list):
                raise ValueError(
                    f"Split '{split_name}' in '{self.data_splits_path}' "
                    "must be a list of scene names."
                )
            normalized = ["".join(str(scene).split()) for scene in scenes]
            if len(normalized) != len(set(normalized)):
                raise ValueError(
                    f"Split '{split_name}' in '{self.data_splits_path}' "
                    "contains duplicate scenes."
                )
            normalized_splits[split_name] = set(normalized)

        split_names = list(normalized_splits)
        for i, first_name in enumerate(split_names):
            for second_name in split_names[i + 1 :]:
                overlap = normalized_splits[first_name] & normalized_splits[second_name]
                if overlap:
                    examples = ", ".join(sorted(overlap)[:5])
                    raise ValueError(
                        f"Scene leakage between splits '{first_name}' and "
                        f"'{second_name}' in '{self.data_splits_path}': {examples}"
                    )

    @staticmethod
    def _parse_rotation_angles(angles):
        rotation_ks = []
        for angle in angles:
            angle = int(angle) % 360
            if angle % 90 != 0:
                raise ValueError(
                    "map_pose_rot_aug angles must be multiples of 90 degrees."
                )
            rotation_ks.append((angle // 90) % 4)
        if not rotation_ks:
            rotation_ks = [0]
        return rotation_ks

    def _sample_map_pose_rotation(self):
        if (
            self.split != "train"
            or not self.map_pose_rot_aug_enable
            or self.map_pose_rot_aug_p <= 0.0
            or np.random.rand() >= self.map_pose_rot_aug_p
        ):
            return 0
        return int(np.random.choice(self.map_pose_rot_aug_angles))

    @staticmethod
    def _rotate_array_90(array, rotation_k):
        rotation_k = int(rotation_k) % 4
        if rotation_k == 0:
            return array
        return np.ascontiguousarray(np.rot90(array, k=rotation_k))

    @staticmethod
    def _rotate_pose_wh_90(pose, width, height, rotation_k):
        rotation_k = int(rotation_k) % 4
        pose = np.asarray(pose, dtype=np.float32).copy()
        x, y, theta = pose
        if rotation_k == 1:
            pose[0] = y
            pose[1] = width - 1.0 - x
            pose[2] = theta - np.pi / 2.0
            new_width, new_height = height, width
        elif rotation_k == 2:
            pose[0] = width - 1.0 - x
            pose[1] = height - 1.0 - y
            pose[2] = theta + np.pi
            new_width, new_height = width, height
        elif rotation_k == 3:
            pose[0] = height - 1.0 - y
            pose[1] = x
            pose[2] = theta + np.pi / 2.0
            new_width, new_height = height, width
        else:
            new_width, new_height = width, height

        pose[0] = np.clip(pose[0], 0.0, max(float(new_width - 1), 0.0))
        pose[1] = np.clip(pose[1], 0.0, max(float(new_height - 1), 0.0))
        pose[2] = np.mod(pose[2], 2.0 * np.pi)
        return pose, int(new_width), int(new_height)

    def _default_map_res(self):
        if self.dataset_type in ("zind", "semrayloc"):
            return 0.01
        return 0.02

    def _get_hard_negative_mode(self):
        hard_negative_cfg = self.dataset_cfg.get("hard_negative", {})
        if isinstance(hard_negative_cfg, dict):
            mode = hard_negative_cfg.get("mode", None)
        else:
            mode = hard_negative_cfg

        mode = self.dataset_cfg.get("hard_negative_mode", mode)
        mode = (mode or "mixed").lower()
        aliases = {
            "pos": "position",
            "trans": "position",
            "translation": "position",
            "ori": "orientation",
            "rot": "orientation",
            "rotation": "orientation",
            "off": "none",
            "false": "none",
            "disable": "none",
            "disabled": "none",
        }
        mode = aliases.get(mode, mode)
        if mode not in ("mixed", "position", "orientation", "none"):
            raise ValueError(
                "Unsupported hard negative mode "
                f"'{mode}'. Expected one of: mixed, position, orientation, none."
            )
        return mode

    def _scene_format(self, scene_dir):
        pose_theta_sign = 1.0
        if self.dataset_type in ("s3d", "structured3d"):
            pose_in_meters = False
            pose_file = "poses_map.txt"
            rgb_dir = "imgs"
            map_res = 0.02
            map_file = "map.png"
            pose_meter_origin = "center"
        elif self.dataset_type == "zind":
            # prepare_zind.py writes map-aligned pixel poses to a 1 cm/pixel map.
            pose_in_meters = False
            pose_file = "poses_map.txt"
            rgb_dir = "rgb"
            map_res = 0.01
            map_file = "map.png"
            pose_meter_origin = "center"
        elif self.dataset_type == "semrayloc":
            pose_in_meters = True
            pose_file = "poses.txt"
            rgb_dir = "rgb"
            map_res = 0.01
            map_file = "floorplan_semantic.png"
            pose_meter_origin = "top_left"
            pose_theta_sign = -1.0
        elif os.path.exists(os.path.join(scene_dir, "poses_map.txt")):
            pose_in_meters = False
            pose_file = "poses_map.txt"
            rgb_dir = "imgs"
            map_res = 0.02
            map_file = "map.png"
            pose_meter_origin = "center"
        else:
            pose_in_meters = True
            pose_file = "poses.txt"
            rgb_dir = "rgb"
            map_res = 0.01
            map_file = "map.png"
            pose_meter_origin = "center"

        return {
            "pose_file": self.dataset_cfg.get("pose_file", pose_file),
            "rgb_dir": self.dataset_cfg.get("rgb_dir", rgb_dir),
            "pose_in_meters": self.dataset_cfg.get("pose_in_meters", pose_in_meters),
            "map_res": float(self.dataset_cfg.get("map_res", map_res)),
            "map_file": self.dataset_cfg.get("map_file", map_file),
            "pose_meter_origin": self.dataset_cfg.get(
                "pose_meter_origin", pose_meter_origin
            ),
            "pose_theta_sign": float(
                self.dataset_cfg.get("pose_theta_sign", pose_theta_sign)
            ),
            "pose_theta_offset": float(self.dataset_cfg.get("pose_theta_offset", 0.0)),
        }

    @staticmethod
    def _sort_key(path: Path):
        stem = path.stem
        if "-" in stem:
            major, minor = stem.split("-", 1)
            return (int(major), int(minor))
        return (int(stem), 0)

    @staticmethod
    def _convert_meter_poses_to_pixels(
        pose_data,
        map_path,
        map_res,
        pose_meter_origin="center",
    ):
        with Image.open(map_path) as map_img:
            map_w, map_h = map_img.size

        pose_meter_origin = pose_meter_origin.lower()
        if pose_meter_origin not in ("center", "top_left"):
            raise ValueError(
                "pose_meter_origin must be one of {'center', 'top_left'}, "
                f"got '{pose_meter_origin}'."
            )

        for pose in pose_data:
            pose[0] = pose[0] / map_res
            pose[1] = pose[1] / map_res
            if pose_meter_origin == "center":
                pose[0] += map_w / 2
                pose[1] += map_h / 2
        return pose_data

    @staticmethod
    def _convert_pose_theta(pose_data, theta_sign=1.0, theta_offset=0.0):
        for pose in pose_data:
            pose[2] = (theta_sign * pose[2] + theta_offset) % (2.0 * np.pi)
        return pose_data

    def _load_data(self, data_folder, data_split):
        data = []
        missing_scenes = []
        map_data_folder = self.dataset_cfg.get("map_data_folder")
        for scene in data_split:
            cur_dir = os.path.join(data_folder, scene)
            if not os.path.isdir(cur_dir):
                missing_scenes.append(scene)
                continue

            scene_format = self._scene_format(cur_dir)
            map_dir = (
                os.path.join(map_data_folder, scene) if map_data_folder else cur_dir
            )
            map_path = os.path.join(map_dir, scene_format["map_file"])
            pose_path = os.path.join(cur_dir, scene_format["pose_file"])
            depth_path = os.path.join(
                cur_dir,
                self.dataset_cfg.get("depth_file", "depth40.txt"),
            )
            rgb_dir = os.path.join(cur_dir, scene_format["rgb_dir"])
            required_paths = {
                "floorplan": map_path,
                "pose file": pose_path,
                "depth file": depth_path,
                "RGB directory": rgb_dir,
            }
            for path_type, path in required_paths.items():
                exists = (
                    os.path.isdir(path)
                    if path_type.endswith("directory")
                    else os.path.isfile(path)
                )
                if not exists:
                    raise FileNotFoundError(
                        f"Scene '{scene}' is missing its {path_type}: '{path}'."
                    )

            pose_data = [
                list(map(float, line.split()))[:3]
                for line in Path(pose_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if scene_format["pose_in_meters"]:
                pose_data = self._convert_meter_poses_to_pixels(
                    pose_data,
                    map_path,
                    scene_format["map_res"],
                    scene_format["pose_meter_origin"],
                )
            pose_data = self._convert_pose_theta(
                pose_data,
                scene_format["pose_theta_sign"],
                scene_format["pose_theta_offset"],
            )

            ray_data = [
                list(map(float, line.split()))
                for line in Path(depth_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            files = sorted(
                (
                    p
                    for p in Path(rgb_dir).iterdir()
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
                ),
                key=self._sort_key,
            )

            counts = {
                "RGB images": len(files),
                "poses": len(pose_data),
                "depth rows": len(ray_data),
            }
            if len(set(counts.values())) != 1:
                count_summary = ", ".join(
                    f"{name}={count}" for name, count in counts.items()
                )
                raise ValueError(
                    f"Scene '{scene}' has mismatched sample counts: {count_summary}."
                )
            if not files:
                raise ValueError(f"Scene '{scene}' contains no samples.")

            for n in range(len(files)):
                data.append(
                    {
                        "rgb_image": str(files[n]),
                        "floorplan_image": map_path,
                        "pose": pose_data[n],
                        "ray": ray_data[n],
                    }
                )
        if missing_scenes:
            examples = ", ".join(missing_scenes[:5])
            suffix = "" if len(missing_scenes) <= 5 else ", ..."
            warnings.warn(
                f"Split '{self.split}' lists {len(missing_scenes)} scene "
                f"directories that are unavailable under '{data_folder}'. "
                f"They were excluded: {examples}{suffix}",
                RuntimeWarning,
            )
        return data

    @staticmethod
    def _build_semantic_onehot_labels(rgb_img):
        rgb = rgb_img.astype(np.int16)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        max_gb = np.maximum(g, b)
        max_rg = np.maximum(r, g)
        mean = rgb.mean(axis=-1)

        # Channel order after one-hot: free, wall, door/opening, window, other.
        door = (r > 120) & ((r - max_gb) > 50)
        window = (b > 120) & ((b - max_rg) > 50)
        wall = (mean < 120) & ~door & ~window
        free = (mean > 220) & ~door & ~window
        other = ~(free | wall | door | window)

        labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
        labels[wall] = 1
        labels[door] = 2
        labels[window] = 3
        labels[other] = 4
        return labels

    def _load_floorplan(self, floorplan_path, rotation_k=0):
        with Image.open(floorplan_path) as img:
            if self.floorplan_representation == "semantic_onehot":
                img = img.convert("RGB")
                rgb = np.asarray(img, dtype=np.uint8)
                rgb = self._rotate_array_90(rgb, rotation_k)
                labels = self._build_semantic_onehot_labels(rgb)
                label_img = Image.fromarray(labels, mode="L")
                label_img = label_img.resize(
                    self.floorplan_img_size,
                    resample=Image.Resampling.NEAREST,
                )
                labels = np.asarray(label_img, dtype=np.int64)
                onehot = np.eye(5, dtype=np.float32)[labels]
                return torch.from_numpy(onehot).permute(2, 0, 1).contiguous()

            img = img.convert("RGB")
            rgb = np.asarray(img, dtype=np.uint8)
            rgb = self._rotate_array_90(rgb, rotation_k)
            img = Image.fromarray(rgb, mode="RGB")
            img = img.resize(self.floorplan_img_size)
            return transforms.ToTensor()(img)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        data = self.data[i]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        with Image.open(data["rgb_image"]) as image:
            rgb_image = transform(image.convert("RGB"))

        pose = torch.tensor(data["pose"])
        ray = torch.tensor(data["ray"])

        with Image.open(data["floorplan_image"]) as image:
            w, h = image.size
        rotation_k = self._sample_map_pose_rotation()
        if rotation_k != 0:
            pose_np, w, h = self._rotate_pose_wh_90(
                pose.numpy(),
                w,
                h,
                rotation_k,
            )
            pose = torch.from_numpy(pose_np)
        wh_tensor = torch.tensor([w, h], dtype=torch.float32)
        floorplan_img = self._load_floorplan(data["floorplan_image"], rotation_k)

        if self.local_map_representation == "semantic_onehot":
            with Image.open(data["floorplan_image"]) as image:
                raw_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            raw_rgb = self._rotate_array_90(raw_rgb, rotation_k)
            raw_map = self._build_semantic_onehot_labels(raw_rgb)
        else:
            raw_map = cv2.imread(data["floorplan_image"], cv2.IMREAD_GRAYSCALE)
            if raw_map is None:
                raise RuntimeError(
                    f"OpenCV failed to decode floorplan '{data['floorplan_image']}'."
                )
            raw_map = self._rotate_array_90(raw_map, rotation_k)
        pose_aug = pose.numpy().copy()
        if self.split == "train" and self.pose_aug_params.get("enable", False):
            trans_range = self.pose_aug_params.get("trans_range", 25)
            rot_range = self.pose_aug_params.get("rot_range", 0.26)
            pose_aug[0] += np.random.uniform(-trans_range, trans_range)
            pose_aug[1] += np.random.uniform(-trans_range, trans_range)
            pose_aug[2] += np.random.uniform(-rot_range, rot_range)

        crop_size_meters = self.dataset_cfg.get("local_map_crop_size_meters", 5.0)
        local_map = self.crop_local_map_tensor(raw_map, pose_aug, crop_size_meters)

        if self.hard_negative_mode == "none":
            neg_pose = torch.zeros(3, dtype=torch.float32)
            neg_local_map = torch.zeros_like(local_map)
        else:
            neg_pose_list = self.get_hard_negative_pose(pose.numpy())
            neg_pose = torch.tensor(neg_pose_list, dtype=torch.float32)
            neg_local_map = self.crop_local_map_tensor(
                raw_map,
                neg_pose.numpy(),
                crop_size_meters,
            )

        return (
            torch.as_tensor(rgb_image, dtype=torch.float32),
            torch.as_tensor(pose, dtype=torch.float32),
            torch.as_tensor(ray, dtype=torch.float32),
            torch.as_tensor(floorplan_img, dtype=torch.float32),
            torch.as_tensor(wh_tensor, dtype=torch.float32),
            local_map,
            neg_local_map,
            torch.as_tensor(neg_pose, dtype=torch.float32),
        )

    def get_hard_negative_pose(self, pose):
        x, y, theta = pose
        if self.hard_negative_mode == "orientation" or (
            self.hard_negative_mode == "mixed" and np.random.rand() < 0.5
        ):
            theta_new = theta + np.pi + np.random.uniform(-0.2, 0.2)
            return [x, y, theta_new]

        dist_m = np.random.uniform(1.5, 3.0)
        dist_px = dist_m / self.map_res
        angle = np.random.uniform(0, 2 * np.pi)
        x_new = x + dist_px * np.cos(angle)
        y_new = y + dist_px * np.sin(angle)
        theta_new = theta + np.random.uniform(-0.2, 0.2)
        return [x_new, y_new, theta_new]

    def crop_local_map(
        self,
        map_img,
        pose,
        crop_size_meters,
        output_size=128,
        interpolation=cv2.INTER_LINEAR,
        resize_interpolation=cv2.INTER_AREA,
        border_value=255,
    ):
        x, y, theta = pose
        crop_size_px = int(crop_size_meters / self.map_res)

        h, w = map_img.shape
        pad = crop_size_px
        map_padded = cv2.copyMakeBorder(
            map_img,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )

        center = (x + pad, y + pad)
        angle_deg = np.degrees(theta)
        rot_matrix = cv2.getRotationMatrix2D(center, angle_deg + 90, 1.0)
        rot_matrix[0, 2] += (crop_size_px / 2.0) - center[0]
        rot_matrix[1, 2] += (crop_size_px / 2.0) - center[1]

        local_map = cv2.warpAffine(
            map_padded,
            rot_matrix,
            (crop_size_px, crop_size_px),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

        if crop_size_px != output_size:
            local_map = cv2.resize(
                local_map,
                (output_size, output_size),
                interpolation=resize_interpolation,
            )

        return local_map

    def crop_local_map_tensor(
        self,
        map_img,
        pose,
        crop_size_meters,
        output_size=128,
    ):
        if self.local_map_representation == "semantic_onehot":
            labels = self.crop_local_map(
                map_img,
                pose,
                crop_size_meters,
                output_size=output_size,
                interpolation=cv2.INTER_NEAREST,
                resize_interpolation=cv2.INTER_NEAREST,
                border_value=4,
            ).astype(np.int64)
            labels = np.clip(labels, 0, 4)
            onehot = np.eye(5, dtype=np.float32)[labels]
            return torch.from_numpy(onehot).permute(2, 0, 1).contiguous()

        local_map = self.crop_local_map(
            map_img,
            pose,
            crop_size_meters,
            output_size=output_size,
        )
        return torch.from_numpy(local_map).float().unsqueeze(0) / 255.0
