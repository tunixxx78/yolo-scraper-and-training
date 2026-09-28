from pathlib import Path
import argparse
import shutil
from ultralytics import YOLO


def dataset_run_name(dataset_yaml: Path, base_model: str) -> str:
    dataset_name = dataset_yaml.parent.name
    model_name = Path(base_model).stem
    return f"{dataset_name}_{model_name}"


def train_model(args):
    dataset_yaml = Path(args.dataset).expanduser().resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    run_name = args.name or dataset_run_name(dataset_yaml, args.model)
    model = YOLO(args.model)

    print(f"Dataset:    {dataset_yaml}")
    print(f"Base model: {args.model}")
    print(f"Run name:   {run_name}")

    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=run_name,
    )

    # Ultralytics exposes the actual save directory through the trainer.
    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    print(f"\nTraining finished.")
    print(f"Best model: {best_pt}")
    print(f"Last model: {last_pt}")

    if args.no_onnx:
        return

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt was not created: {best_pt}")

    export_onnx(
        best_pt,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        device=args.export_device,
    )


def export_onnx(model_path, imgsz=640, opset=None, simplify=True,
                dynamic=False, device="cpu"):
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nExporting ONNX from: {model_path}")
    model = YOLO(str(model_path))

    kwargs = {
        "format": "onnx",
        "imgsz": imgsz,
        "simplify": simplify,
        "dynamic": dynamic,
        "device": device,
        "batch": 1,
    }
    if opset is not None:
        kwargs["opset"] = opset

    exported = Path(model.export(**kwargs)).resolve()

    # Keep deployment files together with the trained weights.
    target = model_path.parent / f"{model_path.stem}.onnx"
    if exported != target:
        shutil.copy2(exported, target)

    print(f"ONNX model: {target}")
    return target


def resume_training(args):
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print(f"Resuming the SAME interrupted training run from: {model_path}")
    model = YOLO(str(model_path))
    model.train(resume=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train an Ultralytics YOLO model and export best.pt to ONNX."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Start a new training run.")
    p_train.add_argument("dataset", help="Path to dataset.yaml")
    p_train.add_argument("--model", default="yolo26n.pt",
                         help="Base model (default: yolo26n.pt)")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--device", default=None,
                         help="Training device, e.g. mps, 0, cpu. Default lets Ultralytics choose.")
    p_train.add_argument("--project", default="models")
    p_train.add_argument("--name", default=None)
    p_train.add_argument("--no-onnx", action="store_true",
                         help="Do not export best.pt to ONNX after training.")
    p_train.add_argument("--opset", type=int, default=None,
                         help="ONNX opset. Default lets Ultralytics choose a supported opset.")
    p_train.add_argument("--no-simplify", action="store_true")
    p_train.add_argument("--dynamic", action="store_true",
                         help="Export ONNX with dynamic input dimensions.")
    p_train.add_argument("--export-device", default="cpu",
                         help="Device used for ONNX export (default: cpu).")
    p_train.set_defaults(func=train_model)

    p_export = sub.add_parser("export", help="Export an existing .pt model to ONNX.")
    p_export.add_argument("model_path", help="Path to best.pt or another .pt checkpoint")
    p_export.add_argument("--imgsz", type=int, default=640)
    p_export.add_argument("--opset", type=int, default=None)
    p_export.add_argument("--no-simplify", action="store_true")
    p_export.add_argument("--dynamic", action="store_true")
    p_export.add_argument("--device", default="cpu")
    p_export.set_defaults(
        func=lambda a: export_onnx(
            a.model_path, a.imgsz, a.opset, not a.no_simplify, a.dynamic, a.device
        )
    )

    p_resume = sub.add_parser(
        "resume",
        help="Resume the SAME interrupted Ultralytics training run from last.pt."
    )
    p_resume.add_argument("model_path", help="Path to last.pt")
    p_resume.set_defaults(func=resume_training)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
