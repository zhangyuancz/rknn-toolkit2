# rknn_yolov26_demo

YOLOv26 / YOLOv8 RKNN 推理 Demo（aarch64 Linux）

## 目录结构

```
rknn_yolov26_demo/
├── CMakeLists.txt
├── build-linux.sh          # 交叉编译脚本
├── include/
│   └── postprocess.h       # YOLOv8-style 后处理接口
├── src/
│   ├── main.cc             # 主推理流程（含 warm-up 和 benchmark）
│   └── postprocess.cc      # NMS、坐标解码、COCO 80 类标签
└── model/                  # 模型和测试图片（由 build-linux.sh -m/-i 自动填充）
    ├── *.rknn
    └── *.jpg
```

## 快速开始

### 1. 转换模型（在 x86 主机上）

```bash
conda activate rknn
cd rknn-toolkit2/examples/onnx/yolov26
python convert_yolov26.py --onnx best.onnx --output yolov26_int8.rknn --target rk3588
```

### 2. 交叉编译（在 x86 主机上）

```bash
cd rknpu2/examples/rknn_yolov26_demo
GCC_COMPILER=aarch64-linux-gnu bash build-linux.sh \
    -t rk3588 \
    -m /path/to/yolov26_int8.rknn \
    -i /path/to/bus.jpg
```

编译产物位于 `install/rknn_yolov26_demo_Linux/`。

### 3. 部署到设备

```bash
scp -r install/rknn_yolov26_demo_Linux pi@<device_ip>:~/
```

### 4. 在设备上运行

```bash
cd ~/rknn_yolov26_demo_Linux
LD_LIBRARY_PATH=./lib ./rknn_yolov26_demo model/yolov26_int8.rknn model/bus.jpg result.jpg
```

## 支持平台

| --target  | 芯片            |
|-----------|-----------------|
| rk3588    | RK3588 / RK3588S |
| rk3576    | RK3576          |
| rk3568    | RK3568          |
| rk3566    | RK3566          |
| rk3562    | RK3562          |

## 性能参考（NanoPi R6C / RK3588S）

- 单次推理：~24 ms
- 平均推理（20次 benchmark）：~22 ms（≈45 FPS）
- 量化：INT8，输出头已拆分为独立 box/class 张量以保证量化精度
