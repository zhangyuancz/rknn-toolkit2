#include "postprocess.h"

#include <math.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <numeric>

// COCO 80-class labels
const char *COCO_LABELS[OBJ_CLASS_NUM] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   cls_id;
};

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float iou(const Detection &a, const Detection &b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua <= 0.f ? 0.f : inter / ua;
}

static void nms(std::vector<Detection> &dets, float thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b){ return a.score > b.score; });
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[i].cls_id == dets[j].cls_id &&
                iou(dets[i], dets[j]) > thresh)
                suppressed[j] = true;
        }
    }
    std::vector<Detection> kept;
    for (size_t i = 0; i < dets.size(); ++i)
        if (!suppressed[i]) kept.push_back(dets[i]);
    dets = kept;
}

void yolov26_post_process(const int8_t *boxes_i8,
                          int32_t box_zp, float box_scale,
                          const int8_t *classes_i8,
                          int32_t cls_zp, float cls_scale,
                          int pad_left, int pad_top,
                          float scale_w, float scale_h,
                          int img_w, int img_h,
                          float conf_thresh, float nms_thresh,
                          detect_result_group_t *group)
{
    group->count = 0;

    // boxes_i8   layout: [4, 8400]   rows=cx,cy,w,h  cols=predictions
    // classes_i8 layout: [80, 8400]  rows=class_idx  cols=predictions
    const int N = MODEL_OUT_COLS;   // 8400

    std::vector<Detection> dets;
    dets.reserve(256);

    for (int j = 0; j < N; ++j) {
        // Find best class (already sigmoid'd probabilities)
        int   best_cls  = -1;
        float best_prob = -1.f;
        for (int c = 0; c < OBJ_CLASS_NUM; ++c) {
            float p = ((float)classes_i8[c * N + j] - cls_zp) * cls_scale;
            if (p > best_prob) { best_prob = p; best_cls = c; }
        }

        if (best_prob < conf_thresh) continue;

        // Dequantise box coords (cx, cy, w, h) in model-input pixel space
        float cx = ((float)boxes_i8[0 * N + j] - box_zp) * box_scale;
        float cy = ((float)boxes_i8[1 * N + j] - box_zp) * box_scale;
        float bw = ((float)boxes_i8[2 * N + j] - box_zp) * box_scale;
        float bh = ((float)boxes_i8[3 * N + j] - box_zp) * box_scale;

        // Remove letterbox padding, map back to original image coords
        float x1 = (cx - bw * 0.5f - pad_left) / scale_w;
        float y1 = (cy - bh * 0.5f - pad_top)  / scale_h;
        float x2 = (cx + bw * 0.5f - pad_left) / scale_w;
        float y2 = (cy + bh * 0.5f - pad_top)  / scale_h;

        x1 = clampf(x1, 0.f, (float)(img_w - 1));
        y1 = clampf(y1, 0.f, (float)(img_h - 1));
        x2 = clampf(x2, 0.f, (float)(img_w - 1));
        y2 = clampf(y2, 0.f, (float)(img_h - 1));

        if (x2 <= x1 || y2 <= y1) continue;

        Detection d;
        d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
        d.score = best_prob;
        d.cls_id = best_cls;
        dets.push_back(d);
    }

    nms(dets, nms_thresh);

    int n = (int)dets.size();
    if (n > OBJ_NUMB_MAX_SIZE) n = OBJ_NUMB_MAX_SIZE;
    group->count = n;
    for (int i = 0; i < n; ++i) {
        detect_result_t *r = &group->results[i];
        strncpy(r->name, COCO_LABELS[dets[i].cls_id], sizeof(r->name) - 1);
        r->name[sizeof(r->name) - 1] = '\0';
        r->box.left   = dets[i].x1;
        r->box.top    = dets[i].y1;
        r->box.right  = dets[i].x2;
        r->box.bottom = dets[i].y2;
        r->prop   = dets[i].score;
        r->cls_id = dets[i].cls_id;
    }
}
