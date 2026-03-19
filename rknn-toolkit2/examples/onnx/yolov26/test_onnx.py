import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from rknn.api import RKNN

try:
    import onnx
except ImportError:
    onnx = None


def _dim_to_int(dim):
    if hasattr(dim, "dim_value") and dim.dim_value > 0:
        return int(dim.dim_value)
    if isinstance(dim, int):
        return int(dim)
    return None


def inspect_onnx_io(onnx_path):
    """Read ONNX IO metadata to detect head style differences between YOLO versions."""
    if onnx is None:
        print("[WARN] onnx package not installed, skip ONNX head inspection.")
        return None

    model = onnx.load(str(onnx_path))
    graph = model.graph

    input_infos = []
    for item in graph.input:
        shape = []
        tensor_type = item.type.tensor_type
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                shape.append(_dim_to_int(d))
        input_infos.append((item.name, shape))

    output_infos = []
    for item in graph.output:
        shape = []
        tensor_type = item.type.tensor_type
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                shape.append(_dim_to_int(d))
        output_infos.append((item.name, shape))

    print("[INFO] ONNX inputs:")
    for name, shape in input_infos:
        print(f"  - {name}: {shape}")

    print("[INFO] ONNX outputs:")
    for name, shape in output_infos:
        print(f"  - {name}: {shape}")

    return {
        "inputs": input_infos,
        "outputs": output_infos,
    }


def guess_yolo_family(io_meta):
    if io_meta is None:
        return "unknown"

    output_num = len(io_meta["outputs"])
    if output_num >= 3:
        return "yolov5_like"

    if output_num == 1:
        shape = io_meta["outputs"][0][1]
        if len(shape) == 3:
            c1, c2 = shape[1], shape[2]
            if c1 is not None and c1 <= 256:
                return "yolov8_like"
            if c2 is not None and c2 <= 256:
                return "yolov8_like"
        return "single_output_unknown"

    return "unknown"


def create_default_dataset_if_needed(dataset_path, sample_image):
    dataset_path = Path(dataset_path)
    if dataset_path.exists():
        return dataset_path

    if sample_image is None:
        raise FileNotFoundError(
            f"dataset file not found: {dataset_path}. Provide --dataset for quantization."
        )

    sample_image = Path(sample_image)
    if not sample_image.exists():
        raise FileNotFoundError(
            f"dataset file not found: {dataset_path}, and sample image not found: {sample_image}."
        )

    dataset_path.write_text(str(sample_image.resolve()) + "\n", encoding="utf-8")
    print(f"[INFO] dataset file auto-created: {dataset_path.resolve()}")
    return dataset_path


def fix_onnx_toposort(onnx_path, suffix="_toposorted"):
    if onnx is None:
        print("[WARN] onnx package not installed, cannot auto-fix topology.")
        return onnx_path

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    nodes = list(model.graph.node)

    producer = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                producer[out] = idx

    indeg = [0] * len(nodes)
    reverse_edges = defaultdict(list)
    for idx, node in enumerate(nodes):
        deps = set()
        for inp in node.input:
            if not inp:
                continue
            pred = producer.get(inp)
            if pred is not None and pred != idx:
                deps.add(pred)
        indeg[idx] = len(deps)
        for pred in deps:
            reverse_edges[pred].append(idx)

    queue = deque(i for i, deg in enumerate(indeg) if deg == 0)
    order = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in reverse_edges[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(nodes):
        print("[WARN] topo sort failed, keep original ONNX.")
        return onnx_path

    fixed_path = onnx_path.with_name(f"{onnx_path.stem}{suffix}{onnx_path.suffix}")
    del model.graph.node[:]
    model.graph.node.extend([nodes[i] for i in order])
    onnx.save(model, str(fixed_path))

    try:
        onnx.checker.check_model(onnx.load(str(fixed_path)))
        print(f"[INFO] topologically sorted ONNX saved: {fixed_path.resolve()}")
        return fixed_path
    except Exception as err:
        print(f"[WARN] fixed ONNX checker failed: {err}")
        return onnx_path


def run_rknn_convert(args):
    onnx_path = Path(args.onnx)
    rknn_path = Path(args.output)

    if not onnx_path.exists():
        raise FileNotFoundError(f"onnx model not found: {onnx_path}")

    if args.auto_fix_toposort:
        onnx_path = fix_onnx_toposort(onnx_path)

    io_meta = inspect_onnx_io(onnx_path)
    detected_family = guess_yolo_family(io_meta)
    print(f"[INFO] detected yolo head family: {detected_family}")

    # Version-aware guardrails:
    # - YOLOv5-like usually exports 3 feature maps (anchor-based).
    # - YOLOv26 (Ultralytics family) is typically YOLOv8-like (anchor-free single output).
    if args.yolo_version in ("yolov26", "yolov8", "yolo11") and detected_family == "yolov5_like":
        print("[WARN] version says yolov26/yolov8-like, but ONNX outputs look yolov5-like.")
        print("[WARN] check export args, especially end2end and model source.")
    if args.yolo_version == "yolov5" and detected_family == "yolov8_like":
        print("[WARN] version says yolov5, but ONNX outputs look yolov8/yolov26-like.")

    rknn = RKNN(verbose=args.verbose)

    print("--> Config RKNN")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target,
    )
    print("done")

    print("--> Load ONNX")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError("load_onnx failed")
    print("done")

    if args.quantize:
        dataset_path = create_default_dataset_if_needed(args.dataset, args.calib_image)
        print("--> Build RKNN (INT8 quantization)")
        ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
    else:
        print("--> Build RKNN (FP)")
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        raise RuntimeError("build failed")
    print("done")

    print("--> Export RKNN")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError("export_rknn failed")
    print(f"done: {rknn_path.resolve()}")

    if args.sanity_image:
        print("--> Init runtime for sanity check")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("init_runtime failed")

        img = cv2.imread(args.sanity_image)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {args.sanity_image}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.img_size, args.img_size))
        img = np.expand_dims(img, 0)

        print("--> Run sanity inference")
        outputs = rknn.inference(inputs=[img], data_format=["nhwc"])
        print(f"[INFO] output tensor count: {len(outputs)}")
        for i, out in enumerate(outputs):
            arr = np.asarray(out)
            print(
                f"  - out[{i}] shape={arr.shape}, "
                f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
            )

    rknn.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert YOLO ONNX to RKNN with yolov26-aware checks."
    )
    parser.add_argument("--onnx", default="./best.onnx", help="input onnx path")
    parser.add_argument("--output", default="./best.rknn", help="output rknn path")
    parser.add_argument(
        "--yolo-version",
        default="yolov26",
        choices=["yolov5", "yolov8", "yolo11", "yolov26", "auto"],
        help="model family hint for version-difference checks",
    )
    parser.add_argument("--target", default="rk3588", help="target platform, e.g. rk3566/rk3588")
    parser.add_argument("--quantize", action="store_true", help="enable INT8 quantization")
    parser.add_argument("--dataset", default="./dataset.txt", help="dataset list for quantization")
    parser.add_argument(
        "--calib-image",
        default="./bus.jpg",
        help="auto-create dataset with this image if --dataset missing",
    )
    parser.add_argument(
        "--sanity-image",
        default="./bus.jpg",
        help="optional image for runtime sanity inference, set empty to skip",
    )
    parser.add_argument("--img-size", type=int, default=640, help="inference image size")
    parser.add_argument(
        "--auto-fix-toposort",
        action="store_true",
        default=True,
        help="auto-fix ONNX node order if graph is not topologically sorted",
    )
    parser.add_argument(
        "--no-auto-fix-toposort",
        action="store_false",
        dest="auto_fix_toposort",
        help="disable ONNX topology auto-fix",
    )
    parser.add_argument("--verbose", action="store_true", help="enable RKNN verbose logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sanity_image == "":
        args.sanity_image = None
    run_rknn_convert(args)
