#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t <soc>      目标 SoC (default: rk3588)"
    echo "                支持: rk3562 rk3566 rk3568 rk3576 rk3588 rv1126b"
    echo "  -b <type>     构建类型 Debug|Release (default: Release)"
    echo "  -m <model>    RKNN 模型文件路径，自动拷贝到 model/ 目录"
    echo "  -i <image>    测试图片路径，自动拷贝到 model/ 目录"
    echo ""
    echo "Example:"
    echo "  $0 -t rk3588 -m ../../rknn-toolkit2/examples/onnx/yolov26/yolov26_int8_rk3588.rknn -i ../../rknn-toolkit2/examples/onnx/yolov26/bus.jpg"
    exit 0
}

TARGET_SOC=rk3588
BUILD_TYPE=Release
MODEL_FILE=""
IMAGE_FILE=""

while getopts ":t:b:m:i:h" opt; do
    case $opt in
        t) TARGET_SOC=$OPTARG ;;
        b) BUILD_TYPE=$OPTARG ;;
        m) MODEL_FILE=$OPTARG ;;
        i) IMAGE_FILE=$OPTARG ;;
        h) usage ;;
        :) echo "Option -$OPTARG requires an argument."; exit 1 ;;
        ?) echo "Invalid option: -$OPTARG"; exit 1 ;;
    esac
done

GCC_PREFIX=${GCC_COMPILER:-aarch64-linux-gnu}
export CC=${GCC_PREFIX}-gcc
export CXX=${GCC_PREFIX}-g++

ROOT_PWD=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${ROOT_PWD}/build/linux_aarch64_${BUILD_TYPE}

echo "==================================="
echo "TARGET_SOC = ${TARGET_SOC}"
echo "BUILD_TYPE = ${BUILD_TYPE}"
echo "CC         = ${CC}"
echo "CXX        = ${CXX}"
echo "BUILD_DIR  = ${BUILD_DIR}"
echo "==================================="

# ── 拷贝模型和图片到 model/ 目录 ─────────────────────────────────────────────
mkdir -p "${ROOT_PWD}/model"

if [ -n "${MODEL_FILE}" ]; then
    if [ ! -f "${MODEL_FILE}" ]; then
        echo "[ERROR] 模型文件不存在: ${MODEL_FILE}"
        exit 1
    fi
    cp -v "${MODEL_FILE}" "${ROOT_PWD}/model/"
fi

if [ -n "${IMAGE_FILE}" ]; then
    if [ ! -f "${IMAGE_FILE}" ]; then
        echo "[ERROR] 图片文件不存在: ${IMAGE_FILE}"
        exit 1
    fi
    cp -v "${IMAGE_FILE}" "${ROOT_PWD}/model/"
fi

# 检查 model/ 目录是否有 rknn 文件
RKNN_COUNT=$(find "${ROOT_PWD}/model" -name "*.rknn" 2>/dev/null | wc -l)
if [ "${RKNN_COUNT}" -eq 0 ]; then
    echo "[WARN] model/ 目录下没有 .rknn 文件，install 时不会打包模型。"
    echo "       可通过 -m <model.rknn> 指定模型路径。"
fi

# ── 编译 ──────────────────────────────────────────────────────────────────────
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${ROOT_PWD}" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

make -j$(nproc)
make install

INSTALL_DIR="${ROOT_PWD}/install/rknn_yolov26_demo_Linux"
echo ""
echo "==================================="
echo "安装目录: ${INSTALL_DIR}"
echo "目录结构:"
find "${INSTALL_DIR}" -type f | sort | sed "s|${INSTALL_DIR}/||"
echo "==================================="
