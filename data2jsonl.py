import os
import json
import argparse
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg", ".mpg"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--type", required=True, choices=["image", "video"])
    parser.add_argument("--output_jsonl", required=True, type=str)
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="指定文件后缀，如 .jpg / jpg / .mp4；默认 None 表示使用内置类型后缀集合"
    )
    args = parser.parse_args()

    if args.type == "image":
        exts = IMAGE_EXTS
        prompt = "<image>Is this image real or fake? Answer only Real or Fake"
        media_key = "images"
    else:
        exts = VIDEO_EXTS
        prompt = "<video>Is this video real or fake? Answer only Real or Fake"
        media_key = "videos"

    if args.suffix is not None:
        suffix = args.suffix.lower()
        if not suffix.startswith("."):
            suffix = "." + suffix
        exts = {suffix}

    files = [
        os.path.join(args.input_dir, f)
        for f in sorted(os.listdir(args.input_dir))
        if os.path.isfile(os.path.join(args.input_dir, f))
        and os.path.splitext(f)[1].lower() in exts
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for path in tqdm(files, desc=f"Processing {args.type}s"):
            item = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                media_key: [path]
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
