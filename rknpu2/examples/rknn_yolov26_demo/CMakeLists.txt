cmake_minimum_required(VERSION 3.10)
project(rknn_yolov26_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# ── Architecture ─────────────────────────────────────────────────────────────
if(CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

# ── RKNN Runtime ─────────────────────────────────────────────────────────────
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/Linux/librknn_api)
set(RKNN_RT_LIB  ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)

# ── OpenCV ───────────────────────────────────────────────────────────────────
if(LIB_ARCH STREQUAL "armhf")
  set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
else()
  set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
endif()
find_package(OpenCV REQUIRED)

# ── Executable ───────────────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_yolov26_demo
  src/main.cc
  src/postprocess.cc
)

target_link_libraries(rknn_yolov26_demo
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
)

# ── Install ───────────────────────────────────────────────────────────────────
set(INSTALL_DIR ${CMAKE_SOURCE_DIR}/install/rknn_yolov26_demo_Linux)
set(CMAKE_INSTALL_PREFIX ${INSTALL_DIR})
set(CMAKE_INSTALL_RPATH "lib")

install(TARGETS rknn_yolov26_demo DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)

# Install model files (*.rknn) and test images (*.jpg / *.png) from model/
file(GLOB MODEL_FILES "${CMAKE_SOURCE_DIR}/model/*.rknn")
file(GLOB IMAGE_FILES "${CMAKE_SOURCE_DIR}/model/*.jpg"
                      "${CMAKE_SOURCE_DIR}/model/*.png")
if(MODEL_FILES)
  install(FILES ${MODEL_FILES} DESTINATION model)
endif()
if(IMAGE_FILES)
  install(FILES ${IMAGE_FILES} DESTINATION model)
endif()
