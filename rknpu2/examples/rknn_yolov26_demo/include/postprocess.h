#pragma once

#include <stdint.h>
#include <vector>
#include <string>

#define OBJ_CLASS_NUM     80
#define OBJ_NUMB_MAX_SIZE 64
#define NMS_THRESH        0.45f
#define BOX_THRESH        0.25f

// YOLOv8/YOLOv26 output layout: [84, 8400]
// 84 = 4 (cx,cy,w,h) + 80 (class scores)
// 8400 = 80x80 + 40x40 + 20x20 anchor-free predictions
#define MODEL_OUT_ROWS    84
#define MODEL_OUT_COLS    8400

typedef struct {
    float left, top, right, bottom;
} BOX_RECT_F;

typedef struct {
    char       name[64];
    BOX_RECT_F box;
    float      prop;
    int        cls_id;
} detect_result_t;

typedef struct {
    int              count;
    detect_result_t  results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

/**
 * Post-process split INT8 quantised YOLOv8/YOLOv26 output tensors.
 *
 * Two separate output tensors from the split ONNX model:
 *   boxes_i8   [1, 4, 8400]  – cx, cy, w, h in model-input pixel space (range 0-640)
 *   classes_i8 [1, 80, 8400] – sigmoid class probabilities (range 0-1)
 *
 * Each tensor has its own zp/scale for proper quantization.
 *
 * @param boxes_i8     INT8 box tensor   [4, 8400]
 * @param box_zp       Box quantisation zero-point
 * @param box_scale    Box quantisation scale
 * @param classes_i8   INT8 class tensor [80, 8400]
 * @param cls_zp       Class quantisation zero-point
 * @param cls_scale    Class quantisation scale
 * @param pad_left     Letterbox left padding in model-input pixels
 * @param pad_top      Letterbox top  padding in model-input pixels
 * @param scale_w      Scale factor: model_input_w / orig_img_w
 * @param scale_h      Scale factor: model_input_h / orig_img_h
 * @param img_w        Original image width  (for coordinate clamp)
 * @param img_h        Original image height (for coordinate clamp)
 * @param conf_thresh  Confidence threshold
 * @param nms_thresh   NMS IoU threshold
 * @param group        Output detection results
 */
void yolov26_post_process(const int8_t *boxes_i8,
                          int32_t box_zp, float box_scale,
                          const int8_t *classes_i8,
                          int32_t cls_zp, float cls_scale,
                          int pad_left, int pad_top,
                          float scale_w, float scale_h,
                          int img_w, int img_h,
                          float conf_thresh, float nms_thresh,
                          detect_result_group_t *group);

extern const char *COCO_LABELS[OBJ_CLASS_NUM];
