import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCE_PATH = Path("test_questions.json")
OUTPUT_DIR = Path("eval_sets")
SEED = 20260510
TUNE_RATIO = 0.6
DEV_RATIO = 0.2


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _split_indices(total: int) -> tuple[int, int]:
    tune_count = max(1, int(total * TUNE_RATIO))
    dev_count = max(1, int(total * DEV_RATIO))
    if tune_count + dev_count >= total:
        dev_count = 1
    blind_count = total - tune_count - dev_count
    if blind_count <= 0:
        blind_count = 1
        if tune_count > dev_count:
            tune_count -= 1
        else:
            dev_count -= 1
    return tune_count, dev_count


def build_splits(freeze_date: str | None = None) -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"未找到评测集文件: {SOURCE_PATH}")

    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        grouped[item.get("category", "general")].append(item)

    rng = random.Random(SEED)
    tune_set: list[dict[str, Any]] = []
    dev_set: list[dict[str, Any]] = []
    blind_set: list[dict[str, Any]] = []

    for _, items in grouped.items():
        copied = list(items)
        rng.shuffle(copied)
        tune_count, dev_count = _split_indices(len(copied))
        tune_set.extend(copied[:tune_count])
        dev_set.extend(copied[tune_count:tune_count + dev_count])
        blind_set.extend(copied[tune_count + dev_count:])

    rng.shuffle(tune_set)
    rng.shuffle(dev_set)
    rng.shuffle(blind_set)

    _dump_json(OUTPUT_DIR / "tune_questions.json", tune_set)
    _dump_json(OUTPUT_DIR / "dev_questions.json", dev_set)
    _dump_json(OUTPUT_DIR / "blind_questions.json", blind_set)

    manifest = {
        "source_file": str(SOURCE_PATH),
        "seed": SEED,
        "ratios": {
            "tune": TUNE_RATIO,
            "dev": DEV_RATIO,
            "blind": 1.0 - TUNE_RATIO - DEV_RATIO,
        },
        "counts": {
            "total": len(rows),
            "tune": len(tune_set),
            "dev": len(dev_set),
            "blind": len(blind_set),
        },
        "notes": [
            "采用按 category 分层抽样，保证各类题都进入 tune/dev/blind。",
            "blind_questions.json 建议冻结，不参与提示词或同义词调参。",
        ],
    }

    if freeze_date:
        blind_frozen_path = OUTPUT_DIR / f"blind_questions_{freeze_date}.json"
        manifest_frozen_path = OUTPUT_DIR / f"split_manifest_{freeze_date}.json"
        _dump_json(blind_frozen_path, blind_set)
        manifest["frozen_files"] = {
            "blind_questions": str(blind_frozen_path),
            "split_manifest": str(manifest_frozen_path),
        }
        _dump_json(manifest_frozen_path, manifest)

    _dump_json(OUTPUT_DIR / "split_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建分层 tune/dev/blind 评测集。")
    parser.add_argument(
        "--freeze-date",
        default="",
        help="可选，传入日期(如 2026-05-10)后会额外生成带日期的冻结文件。",
    )
    args = parser.parse_args()
    build_splits(freeze_date=args.freeze_date.strip() or None)
