"""LENS end-to-end CLI.

Usage:
    python run.py --weights yolo26n.pt --image samples/person.jpg
    python run.py --weights yolo26n.pt --image samples/person.jpg --log-only

Every run writes three artifacts into inference/<image_stem>_<timestamp>/:
  - annotated.jpg     YOLO bounding boxes drawn on the input image
  - inference_log.json  the structured log fed to the LLM
  - explanation.txt   the natural-language explanation from the LLM

Set OPENAI_API_KEY in your environment or in a .env file in this folder
(format: OPENAI_API_KEY=sk-...) before running (unless --log-only).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

load_dotenv()

from lens.explainer import LensExplainer, summarize_architecture
from lens.logger import ActivationLogger
from lens.schema import build_inference_log


def pick_top_detection(result) -> dict | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    conf = boxes.conf.cpu().numpy()
    i = int(conf.argmax())
    cls_id = int(boxes.cls[i].item())
    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
    names = result.names
    return {
        "class": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id],
        "class_id": cls_id,
        "confidence": float(conf[i]),
        "bbox": xyxy,
    }


def make_run_dir(image_path: Path, root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{image_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_annotated_image(result, out_path: Path) -> None:
    # result.plot() returns a BGR ndarray with boxes + labels drawn.
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    Image.fromarray(annotated_rgb).save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="LENS — LLM-Enabled Neural Network Summarizer")
    ap.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    ap.add_argument("--output-dir", default="inference", help="Root folder for per-run artifacts")
    ap.add_argument("--log-only", action="store_true", help="Skip LLM call; still save image + JSON")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model id (e.g. gpt-4o, gpt-4o-mini)")
    args = ap.parse_args()

    weights = Path(args.weights)
    image_path = Path(args.image)
    if not weights.exists():
        print(f"weights not found: {weights}", file=sys.stderr)
        return 2
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2

    model = YOLO(str(weights))
    img = Image.open(image_path)
    w, h = img.size

    with ActivationLogger(model) as logger:
        logger.set_image_metadata(str(image_path), w, h)
        results = model.predict(source=str(image_path), conf=args.conf, verbose=False)

    detection = pick_top_detection(results[0])
    nc = getattr(model.model, "nc", None) or len(getattr(model.model, "names", []) or [])
    inference_log = build_inference_log(logger.log, detection, nc=int(nc))

    run_dir = make_run_dir(image_path, Path(args.output_dir))
    save_annotated_image(results[0], run_dir / "annotated.jpg")
    (run_dir / "inference_log.json").write_text(json.dumps(inference_log, indent=2))

    print("=== Detection ===")
    if detection:
        print(f"{detection['class']}  conf={detection['confidence']:.3f}  bbox={[round(v,1) for v in detection['bbox']]}")
    else:
        print("(no detection above threshold)")

    if args.log_only:
        print(f"\nSaved annotated image + JSON log to: {run_dir}")
        return 0

    print("\n=== LENS Explanation ===")
    explainer = LensExplainer(model=args.model)
    arch = summarize_architecture(model)
    # print(f"+++{arch}+++")
    explanation = explainer.explain(inference_log, arch)
    print(explanation)

    (run_dir / "explanation.txt").write_text(explanation)
    print(f"\nSaved annotated image + JSON log + explanation to: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
