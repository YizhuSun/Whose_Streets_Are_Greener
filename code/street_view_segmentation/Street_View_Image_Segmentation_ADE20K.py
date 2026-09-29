  # https://github.com/CSAILVision/sceneparsing
# https://github.com/CSAILVision/semantic-segmentation-pytorch

import csv
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import PIL.Image
import scipy.io
import torch
import torchvision.transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def visualize_result(pred: np.ndarray, colors: np.ndarray, index: int = None) -> np.ndarray:
    """Generate a color-encoded segmentation map from predicted class labels."""
    if index is not None:
        pred = pred.copy()
        pred[  pred != index] = -1
    return colorEncode(pred, colors).astype(np.uint8)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_rel_path(path: str) -> str:
    """
    Normalize relative paths to forward slashes for cross-platform consistency.
    """
    return os.path.normpath(path).replace("\\", "/")


def natural_sort_key(text: str) -> List:
    """
    Return a natural-sort key so numeric filename components are ordered numerically.
    """
    normalized = normalize_rel_path(text).lower()
    parts = re.split(r"(\d+)", normalized)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def atomic_append_csv(csv_path: str, row: List) -> None:
    """
    Append one CSV row and flush it to disk immediately.
    """
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def atomic_save_image(image: PIL.Image.Image, save_path: str) -> None:
    """
    Save an image through a temporary file and atomically replace the target file.
    """
    ensure_dir(os.path.dirname(save_path) or ".")

    base, ext = os.path.splitext(save_path)
    ext = ext.lower()
    save_format_map = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".webp": "WEBP",
    }
    save_format = save_format_map.get(ext, "PNG")

    tmp_path = f"{base}.tmp{ext if ext else '.png'}"
    image.save(tmp_path, format=save_format)
    os.replace(tmp_path, save_path)


def init_result_csv(image_result_csv: str, id_save: int) -> None:
    """
    Initialize the result CSV with a header when it does not yet exist.
    """
    if os.path.exists(image_result_csv):
        return

    headers = ["id"] + [f"id{i + 1}" for i in range(id_save)]
    with open(image_result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        f.flush()
        os.fsync(f.fileno())


def build_basename_index(image_files: List[Tuple[str, str, str]]) -> Dict[str, Set[str]]:
    """
    Build a basename-to-relative-path index for resume compatibility.
    """
    basename_map: Dict[str, Set[str]] = defaultdict(set)
    for _, image_name, rel_path in image_files:
        basename_map[image_name].add(rel_path)
    return basename_map


def resolve_stored_key(stored_key: str, basename_map: Dict[str, Set[str]]) -> Tuple[str, str]:
    """
    Resolve a stored image key to the current relative-path key.

    Returns:
        (resolved_key, status), where status is "ok", "missing", or "ambiguous".
    """
    stored_key = (stored_key or "").strip()
    if not stored_key:
        return "", "missing"

    normalized = normalize_rel_path(stored_key)

    # Current format stores relative paths in CSV/checkpoint files.
    if "/" in normalized:
        return normalized, "ok"

    # Backward compatibility for older basename-only records.
    candidates = sorted(basename_map.get(normalized, set()))
    if not candidates:
        return normalized, "missing"
    if len(candidates) == 1:
        return candidates[0], "ok"
    return normalized, "ambiguous"


def load_processed_from_csv(
    image_result_csv: str,
    basename_map: Dict[str, Set[str]],
) -> Tuple[Set[str], List[str]]:
    """
    Read processed image keys from the result CSV.
    """
    processed: Set[str] = set()
    unresolved: List[str] = []

    if not os.path.exists(image_result_csv):
        return processed, unresolved

    with open(image_result_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_skipped = False
        for row in reader:
            if not row:
                continue
            if not header_skipped:
                header_skipped = True
                if row[0] == "id":
                    continue

            stored_key = row[0].strip()
            resolved_key, status = resolve_stored_key(stored_key, basename_map)
            if status == "ok":
                processed.add(resolved_key)
            elif stored_key:
                unresolved.append(stored_key)

    return processed, unresolved


def load_processed_from_outputs(
    image_result_path: str,
    basename_map: Dict[str, Set[str]],
) -> Tuple[Set[str], List[str]]:
    """
    Read processed image keys from the segmentation-output directory.
    """
    processed: Set[str] = set()
    unresolved: List[str] = []

    if not os.path.isdir(image_result_path):
        return processed, unresolved

    for root, _, files in os.walk(image_result_path):
        for name in files:
            full_path = os.path.join(root, name)
            if not os.path.isfile(full_path):
                continue

            rel_to_result_root = normalize_rel_path(os.path.relpath(full_path, image_result_path))
            resolved_key, status = resolve_stored_key(rel_to_result_root, basename_map)
            if status == "ok":
                processed.add(resolved_key)
            else:
                unresolved.append(rel_to_result_root)

    return processed, unresolved


def load_processed_from_checkpoint(
    checkpoint_path: str,
    basename_map: Dict[str, Set[str]],
) -> Tuple[Set[str], List[str]]:
    """
    Read completed image keys from the checkpoint file.
    """
    processed: Set[str] = set()
    unresolved: List[str] = []

    if not os.path.exists(checkpoint_path):
        return processed, unresolved

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if item.get("status") != "done":
                continue

            stored_key = str(item.get("image_key", "") or item.get("image_name", "")).strip()
            resolved_key, status = resolve_stored_key(stored_key, basename_map)
            if status == "ok":
                processed.add(resolved_key)
            elif stored_key:
                unresolved.append(stored_key)

    return processed, unresolved


def append_checkpoint(checkpoint_path: str, image_key: str, status: str = "done") -> None:
    """
    Append a completed image key to the checkpoint file.
    """
    payload = {
        "image_key": normalize_rel_path(image_key),
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def get_resume_processed_set(
    image_result_csv: str,
    image_result_path: str,
    checkpoint_path: str,
    basename_map: Dict[str, Set[str]],
) -> Set[str]:
    """
    Reconstruct the completed-image set for safe resume after interruption.
    """
    checkpoint_done, checkpoint_unresolved = load_processed_from_checkpoint(checkpoint_path, basename_map)
    csv_done, csv_unresolved = load_processed_from_csv(image_result_csv, basename_map)
    image_done, image_unresolved = load_processed_from_outputs(image_result_path, basename_map)

    fallback_done = csv_done & image_done
    processed = checkpoint_done | fallback_done

    inconsistent_only_csv = sorted(csv_done - image_done)
    inconsistent_only_img = sorted(image_done - csv_done)

    if inconsistent_only_csv:
        print(f"[WARN] 检测到 {len(inconsistent_only_csv)} 条记录仅存在于 CSV，不存在于结果目录，恢复时将重新计算。")
    if inconsistent_only_img:
        print(f"[WARN] 检测到 {len(inconsistent_only_img)} 条记录仅存在于结果目录，不存在于 CSV，恢复时将重新计算。")

    unresolved = checkpoint_unresolved + csv_unresolved + image_unresolved
    if unresolved:
        sample = "；".join(unresolved[:5])
        print(f"[WARN] 检测到 {len(unresolved)} 条历史记录无法唯一恢复（多半是旧版只按文件名记录且存在重名）。这些图片将重新计算。示例: {sample}")

    return processed


def iter_image_files(images_storage_path: str) -> List[Tuple[str, str, str]]:
    """
    Recursively collect input images and return them in stable natural-sort order.
    """
    image_files: List[Tuple[str, str, str]] = []
    images_storage_path = os.path.abspath(images_storage_path)

    for root, _, files in os.walk(images_storage_path):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in VALID_IMAGE_EXTS:
                full_path = os.path.join(root, name)
                rel_path = normalize_rel_path(os.path.relpath(full_path, images_storage_path))
                image_files.append((root, name, rel_path))

    image_files.sort(key=lambda item: natural_sort_key(item[2]))
    return image_files


def build_segmentation_module(device: torch.device) -> SegmentationModule:
    net_encoder = ModelBuilder.build_encoder(
        arch="resnet50dilated",
        fc_dim=2048,
        weights="ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth",
    )
    net_decoder = ModelBuilder.build_decoder(
        arch="ppm_deepsup",
        fc_dim=2048,
        num_class=150,
        weights="ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth",
        use_softmax=True,
    )

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()

    if device.type == "cuda":
        segmentation_module.cuda()
    else:
        segmentation_module.cpu()

    return segmentation_module


def start(images_storage_path, image_result_path, image_result_csv, id_save, color_building):
    start_time = time.time()

    ensure_dir(image_result_path)
    init_result_csv(image_result_csv, id_save)

    checkpoint_path = os.path.splitext(image_result_csv)[0] + "_checkpoint.jsonl"

    # Load segmentation color palette.
    if color_building:
        colors = scipy.io.loadmat("data/color_building.mat")["colors"]
    else:
        colors = scipy.io.loadmat("data/color150.mat")["colors"]

    # Use CUDA when available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[-] 当前推理设备: {device}")

    segmentation_module = build_segmentation_module(device)

    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image_files = iter_image_files(images_storage_path)
    basename_map = build_basename_index(image_files)

    processed_set = get_resume_processed_set(
        image_result_csv=image_result_csv,
        image_result_path=image_result_path,
        checkpoint_path=checkpoint_path,
        basename_map=basename_map,
    )
    print(f"[-] 已检测到 {len(processed_set)} 张已完成图片，将从断点继续执行。")

    total_count = len(image_files)
    print(f"[-] 待检索图片总数: {total_count}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    try:
        for idx, (root, image_name, image_key) in enumerate(image_files, start=1):
            if image_key in processed_set:
                skip_count += 1
                print(f"[{idx}/{total_count}] 跳过已完成图片: {image_key}")
                continue

            image_path = os.path.join(root, image_name)
            save_path = os.path.join(image_result_path, image_key)

            try:
                pil_image = PIL.Image.open(image_path).convert("RGB")
                img_data = pil_to_tensor(pil_image)
                output_size = img_data.shape[1:]

                if device.type == "cuda":
                    singleton_batch = {"img_data": img_data[None].cuda(non_blocking=True)}
                else:
                    singleton_batch = {"img_data": img_data[None].cpu()}

                with torch.no_grad():
                    scores = segmentation_module(singleton_batch, segSize=output_size)

                _, pred = torch.max(scores, dim=1)
                pred = pred.detach().cpu()[0].numpy()

                # Avoid rewriting an existing segmentation image.
                if not os.path.exists(save_path):
                    vs_total = visualize_result(pred, colors, None)
                    atomic_save_image(PIL.Image.fromarray(vs_total), save_path)

                # Compute per-class pixel shares directly from predicted labels.
                pixel_counts = np.bincount(pred.reshape(-1), minlength=max(id_save, int(pred.max()) + 1))
                all_num = pred.size
                rate_list = [image_key]
                for class_id in range(id_save):
                    rate = float(pixel_counts[class_id]) / float(all_num)
                    rate_list.append(rate)

                # Record CSV output before marking the image as complete.
                atomic_append_csv(image_result_csv, rate_list)
                append_checkpoint(checkpoint_path, image_key, status="done")

                processed_set.add(image_key)
                success_count += 1
                print(f"[{idx}/{total_count}] 处理完成: {image_key}")

            except Exception as e:
                fail_count += 1
                print(f"[{idx}/{total_count}] [ERROR] 处理失败: {image_key} | 原因: {e}")
                continue

    except KeyboardInterrupt:
        print("\n[WARN] 检测到手动中断（Ctrl+C），已保留当前已完成进度，可直接重新运行继续。")

    end_time = time.time()
    print("=" * 80)
    print(f"[-] 全部任务结束")
    print(f"[-] 新处理成功: {success_count}")
    print(f"[-] 已跳过: {skip_count}")
    print(f"[-] 失败: {fail_count}")
    print(f"[-] 总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    # Input image directory
    images_storage_path = r"./images"
    # Segmentation output directory
    image_result_path = r"./images_result"
    # Per-class pixel-share CSV
    image_result_csv = r"./images_result_150.csv"
    # Number of classes to report
    id_save = 150
    # Use the building-specific color palette when True
    color_building = False

    start(
        images_storage_path=images_storage_path,
        image_result_path=image_result_path,
        image_result_csv=image_result_csv,
        id_save=id_save,
        color_building=color_building,
    )
