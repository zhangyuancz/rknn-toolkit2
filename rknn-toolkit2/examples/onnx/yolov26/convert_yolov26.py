#!/usr/bin/env python3
"""
convert_yolov26.py  —  一键将 YOLOv26/YOLOv8 ONNX 转换为 RKNN（INT8 量化）

使用方法:
    python convert_yolov26.py --onnx best.onnx --output yolov26_int8.rknn \
        --target rk3588 --calib-image bus.jpg

完整流程:
    1. 拓扑排序修复 ONNX 节点顺序
    2. 拆分最终输出头：去掉混合 [1,84,8400] 输出，
       改为独立的 boxes [1,4,8400] 和 classes [1,80,8400]，
       使两个张量各自获得正确的量化尺度
    3. INT8 量化并导出 RKNN 模型
"""

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

# ── 可选依赖检查 ─────────────────────────────────────────────────────────────
try:
    import onnx
    from onnx import helper, TensorProto, shape_inference as onnx_shape
except ImportError:
    print("[ERROR] onnx 未安装。请执行: pip install onnx")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python 未安装。请执行: pip install opencv-python")
    sys.exit(1)

try:
    from rknn.api import RKNN
except ImportError:
    print("[ERROR] rknn-toolkit2 未安装。请在对应 conda 环境中运行本脚本。")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 拓扑排序
# ─────────────────────────────────────────────────────────────────────────────
def toposort_onnx(src_path: Path) -> Path:
    """对 ONNX 图节点做拓扑排序，返回修复后的文件路径。"""
    model = onnx.load(str(src_path))
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
        print("[WARN] 拓扑排序失败，使用原始 ONNX。")
        return src_path

    dst_path = src_path.with_name(src_path.stem + "_toposorted" + src_path.suffix)
    del model.graph.node[:]
    model.graph.node.extend([nodes[i] for i in order])
    onnx.save(model, str(dst_path))

    try:
        onnx.checker.check_model(onnx.load(str(dst_path)))
        print(f"[Step1] 拓扑排序完成 → {dst_path.name}")
    except Exception as e:
        print(f"[WARN] 拓扑排序后 checker 警告（通常可忽略）: {e}")

    return dst_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 拆分输出头
# ─────────────────────────────────────────────────────────────────────────────
def split_output_heads(src_path: Path) -> Path:
    """
    将 YOLOv8/YOLOv26 的合并输出 [1,84,8400] 拆分为：
      output[0]  boxes   [1, 4, 8400]  — 解码后的 cx,cy,w,h（像素空间）
      output[1]  classes [1,80, 8400]  — sigmoid 类别概率

    这样两个张量各自获得合适的量化尺度：
      boxes   scale ≈ 2.5  （范围 0–640）
      classes scale ≈ 0.004（范围 0–1）
    """
    model = onnx_shape.infer_shapes(onnx.load(str(src_path)))
    graph = model.graph

    # 找最后两个节点：Concat（生成 [1,84,8400]）和其后的 Cast
    # 它们的输出张量名分别是 graph_output_cast_0 和 output0
    remove_outputs = {"graph_output_cast_0", "output0"}

    # 定位两个要导出的张量：最后 Concat 节点的两个输入
    boxes_name = classes_name = None
    for node in reversed(graph.node):
        if node.op_type == "Concat" and any(o in remove_outputs for o in node.output):
            if len(node.input) == 2:
                boxes_name, classes_name = node.input[0], node.input[1]
            break

    if boxes_name is None or classes_name is None:
        # 回退：按形状推断
        all_shapes = {
            v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in list(graph.value_info) + list(graph.input) + list(graph.output)
            if v.type.tensor_type.HasField("shape")
        }
        for node in reversed(graph.node):
            if node.op_type == "Concat":
                inputs_shapes = [all_shapes.get(i) for i in node.input]
                for i, s in zip(node.input, inputs_shapes):
                    if s and s[1] == 4:
                        boxes_name = i
                    if s and s[1] >= 20:
                        classes_name = i
                if boxes_name and classes_name:
                    break

    if boxes_name is None or classes_name is None:
        raise RuntimeError(
            "无法自动定位 box/class 输出张量，请检查 ONNX 图结构。"
        )

    # 移除 Concat 和 Cast 节点
    new_nodes = [n for n in graph.node
                 if not any(o in remove_outputs for o in n.output)]
    removed = len(graph.node) - len(new_nodes)
    del graph.node[:]
    graph.node.extend(new_nodes)

    # 替换图输出
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(boxes_name,   TensorProto.FLOAT, [1, 4,  8400]))
    graph.output.append(
        helper.make_tensor_value_info(classes_name, TensorProto.FLOAT, [1, 80, 8400]))

    dst_path = src_path.with_name(src_path.stem + "_split" + src_path.suffix)
    onnx.save(model, str(dst_path))
    print(f"[Step2] 输出头拆分完成（移除 {removed} 个节点）→ {dst_path.name}")
    print(f"        boxes:   {boxes_name.split('/')[-1]}  [1, 4, 8400]")
    print(f"        classes: {classes_name.split('/')[-1]}  [1, 80, 8400]")
    return dst_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: RKNN 转换（INT8 量化）
# ─────────────────────────────────────────────────────────────────────────────
def convert_to_rknn(onnx_path: Path, output_path: Path,
                    target: str, dataset_path: Path,
                    calib_image: Path, img_size: int,
                    sanity_image: Path | None) -> None:
    rknn = RKNN(verbose=False)

    print(f"[Step3] 配置 RKNN (target={target})")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target,
    )

    print(f"        加载 ONNX: {onnx_path.name}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError("load_onnx 失败")

    # 准备量化校准数据集
    if not dataset_path.exists():
        if calib_image and calib_image.exists():
            dataset_path.write_text(str(calib_image.resolve()) + "\n", encoding="utf-8")
            print(f"        量化数据集自动创建: {dataset_path.name} (使用 {calib_image.name})")
        else:
            raise FileNotFoundError(
                f"量化数据集不存在: {dataset_path}，且 --calib-image 未指定或文件不存在。"
            )

    print("        构建（INT8 量化）...")
    ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
    if ret != 0:
        raise RuntimeError("build 失败")

    print(f"        导出 RKNN → {output_path.name}")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError("export_rknn 失败")

    # 可选：PC 端模拟器 sanity check
    if sanity_image and sanity_image.exists():
        print(f"        Sanity 推理（模拟器）: {sanity_image.name}")
        ret = rknn.init_runtime()
        if ret != 0:
            print("[WARN] init_runtime 失败，跳过 sanity check。")
        else:
            img = cv2.imread(str(sanity_image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = np.expand_dims(img, 0)
            outputs = rknn.inference(inputs=[img], data_format=["nhwc"])
            print(f"        输出张量数: {len(outputs)}")
            for i, out in enumerate(outputs):
                arr = np.asarray(out)
                print(f"          out[{i}] shape={arr.shape}  "
                      f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")

    rknn.release()
    print(f"\n[Done] RKNN 模型已保存: {output_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv26/YOLOv8 ONNX → RKNN INT8 一键转换脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx",         default="./best.onnx",
                   help="输入 ONNX 模型路径")
    p.add_argument("--output",       default="./yolov26_int8.rknn",
                   help="输出 RKNN 模型路径")
    p.add_argument("--target",       default="rk3588",
                   help="目标平台: rk3562 / rk3566 / rk3568 / rk3576 / rk3588")
    p.add_argument("--dataset",      default="./dataset.txt",
                   help="量化校准数据集文件（每行一个图片路径）")
    p.add_argument("--calib-image",  default="./bus.jpg",
                   help="若 --dataset 不存在，自动用此图片创建单张校准集")
    p.add_argument("--img-size",     type=int, default=640,
                   help="模型输入尺寸")
    p.add_argument("--sanity-image", default="./bus.jpg",
                   help="转换后在 PC 模拟器上做 sanity 推理的图片，留空则跳过")
    p.add_argument("--skip-toposort", action="store_true",
                   help="跳过拓扑排序步骤")
    p.add_argument("--skip-split",   action="store_true",
                   help="跳过输出头拆分步骤（仅当 ONNX 已是双输出时使用）")
    return p.parse_args()


def main():
    args = parse_args()

    onnx_path   = Path(args.onnx)
    output_path = Path(args.output)
    calib_image = Path(args.calib_image) if args.calib_image else None
    sanity_img  = Path(args.sanity_image) if args.sanity_image else None

    if not onnx_path.exists():
        print(f"[ERROR] ONNX 文件不存在: {onnx_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"  输入 ONNX : {onnx_path}")
    print(f"  输出 RKNN : {output_path}")
    print(f"  目标平台  : {args.target}")
    print("=" * 60)

    # Step 1: 拓扑排序
    if not args.skip_toposort:
        onnx_path = toposort_onnx(onnx_path)
    else:
        print("[Step1] 跳过拓扑排序")

    # Step 2: 拆分输出头
    if not args.skip_split:
        onnx_path = split_output_heads(onnx_path)
    else:
        print("[Step2] 跳过输出头拆分")

    # Step 3: 转换为 RKNN
    convert_to_rknn(
        onnx_path   = onnx_path,
        output_path = output_path,
        target      = args.target,
        dataset_path= Path(args.dataset),
        calib_image = calib_image,
        img_size    = args.img_size,
        sanity_image= sanity_img,
    )


if __name__ == "__main__":
    main()
