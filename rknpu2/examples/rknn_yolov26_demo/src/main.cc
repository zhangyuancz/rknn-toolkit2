// YOLOv26 (YOLOv8-style) RKNN inference demo
// Target: NanoPi R6C (RK3588S), aarch64 Linux
// Model output: [1, 84, 8400] INT8 quantised
//
// Usage: rknn_yolov26_demo <model.rknn> <image.jpg> [output.jpg]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rknn_api.h"
#include "postprocess.h"

#define MODEL_W 640
#define MODEL_H 640
#define WARMUP_RUNS  3
#define BENCH_RUNS  20

static double get_ms(struct timeval t) {
    return t.tv_sec * 1000.0 + t.tv_usec / 1000.0;
}

static void dump_tensor_attr(rknn_tensor_attr *a) {
    char shape[128] = "";
    for (uint32_t i = 0; i < a->n_dims; ++i) {
        char buf[16];
        snprintf(buf, sizeof(buf), i ? ",%u" : "%u", a->dims[i]);
        strncat(shape, buf, sizeof(shape) - strlen(shape) - 1);
    }
    printf("  [%d] %-20s  shape=[%s]  fmt=%d  type=%d  qnt=%d  zp=%d  scale=%.6f\n",
           a->index, a->name, shape, a->fmt, a->type, a->qnt_type, a->zp, a->scale);
}

// Letterbox resize: scale image into (out_w x out_h) with grey padding.
// Returns: the scale used and (pad_left, pad_top) written to caller.
static float letterbox(const cv::Mat &src, cv::Mat &dst,
                       int out_w, int out_h,
                       int &pad_left, int &pad_top)
{
    float scale = std::min((float)out_w / src.cols, (float)out_h / src.rows);
    int   new_w = (int)(src.cols * scale + 0.5f);
    int   new_h = (int)(src.rows * scale + 0.5f);
    pad_left = (out_w - new_w) / 2;
    pad_top  = (out_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    dst = cv::Mat(out_h, out_w, CV_8UC3, cv::Scalar(128, 128, 128));
    resized.copyTo(dst(cv::Rect(pad_left, pad_top, new_w, new_h)));
    return scale;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s <model.rknn> <image.jpg> [output.jpg]\n", argv[0]);
        return -1;
    }
    const char *model_path  = argv[1];
    const char *image_path  = argv[2];
    std::string output_path = (argc >= 4) ? argv[3] : "result.jpg";

    // ── Load RKNN model ─────────────────────────────────────────────────────
    printf("Loading model: %s\n", model_path);
    FILE *fp = fopen(model_path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open model file\n"); return -1; }
    fseek(fp, 0, SEEK_END);
    long model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *model_data = malloc(model_size);
    fread(model_data, 1, model_size, fp);
    fclose(fp);

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) { fprintf(stderr, "rknn_init failed: %d\n", ret); return -1; }

    // ── Query I/O tensors ────────────────────────────────────────────────────
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("Inputs: %u  Outputs: %u\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        dump_tensor_attr(&input_attrs[i]);
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        dump_tensor_attr(&output_attrs[i]);
    }

    // ── Load & preprocess image ──────────────────────────────────────────────
    printf("Reading image: %s\n", image_path);
    cv::Mat orig = cv::imread(image_path);
    if (orig.empty()) { fprintf(stderr, "Cannot read image\n"); return -1; }
    cv::Mat rgb;
    cv::cvtColor(orig, rgb, cv::COLOR_BGR2RGB);

    cv::Mat model_input;
    int pad_left = 0, pad_top = 0;
    float scale = letterbox(rgb, model_input, MODEL_W, MODEL_H, pad_left, pad_top);
    float scale_w = scale, scale_h = scale;

    printf("Image: %dx%d  ->  model input: %dx%d  pad=(%d,%d)  scale=%.4f\n",
           orig.cols, orig.rows, MODEL_W, MODEL_H, pad_left, pad_top, scale);

    // ── Set RKNN input ───────────────────────────────────────────────────────
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index       = 0;
    inputs[0].type        = RKNN_TENSOR_UINT8;
    inputs[0].fmt         = RKNN_TENSOR_NHWC;
    inputs[0].size        = MODEL_W * MODEL_H * 3;
    inputs[0].buf         = model_input.data;
    inputs[0].pass_through = 0;  // let RKNN normalise (mean=0, std=255)

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        outputs[i].index      = i;
        outputs[i].want_float = 0;  // keep INT8 for speed
    }

    // ── Warm-up ──────────────────────────────────────────────────────────────
    printf("Warming up (%d runs)...\n", WARMUP_RUNS);
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        rknn_outputs_release(ctx, io_num.n_output, outputs);
    }

    // ── Single inference + post-process ─────────────────────────────────────
    struct timeval t0, t1;
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    gettimeofday(&t0, NULL);
    rknn_run(ctx, NULL);
    rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&t1, NULL);
    double infer_ms = get_ms(t1) - get_ms(t0);
    printf("\n[Inference time] %.2f ms\n", infer_ms);

    // Post-process (two outputs: boxes [1,4,8400] and classes [1,80,8400])
    if (io_num.n_output < 2) {
        fprintf(stderr, "Expected 2 output tensors (boxes + classes), got %u\n", io_num.n_output);
        return -1;
    }
    int32_t box_zp  = output_attrs[0].zp;  float box_scale  = output_attrs[0].scale;
    int32_t cls_zp  = output_attrs[1].zp;  float cls_scale  = output_attrs[1].scale;
    printf("  boxes   zp=%d  scale=%.6f\n", box_zp,  box_scale);
    printf("  classes zp=%d  scale=%.6f\n", cls_zp,  cls_scale);

    detect_result_group_t det_grp;
    yolov26_post_process((const int8_t *)outputs[0].buf, box_zp, box_scale,
                         (const int8_t *)outputs[1].buf, cls_zp, cls_scale,
                         pad_left, pad_top,
                         scale_w, scale_h,
                         orig.cols, orig.rows,
                         BOX_THRESH, NMS_THRESH,
                         &det_grp);

    // ── Print results ────────────────────────────────────────────────────────
    printf("\nDetected %d object(s):\n", det_grp.count);
    for (int i = 0; i < det_grp.count; ++i) {
        detect_result_t *r = &det_grp.results[i];
        printf("  [%d] %-16s  conf=%.3f  box=(%.0f,%.0f,%.0f,%.0f)\n",
               i, r->name, r->prop,
               r->box.left, r->box.top, r->box.right, r->box.bottom);
        // Draw on original image
        cv::rectangle(orig,
                      cv::Point((int)r->box.left,  (int)r->box.top),
                      cv::Point((int)r->box.right, (int)r->box.bottom),
                      cv::Scalar(0, 255, 0), 2);
        char label[96];
        snprintf(label, sizeof(label), "%s %.0f%%", r->name, r->prop * 100);
        cv::putText(orig, label,
                    cv::Point((int)r->box.left, (int)r->box.top - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    rknn_outputs_release(ctx, io_num.n_output, outputs);
    cv::imwrite(output_path, orig);
    printf("Result image saved to: %s\n", output_path.c_str());

    // ── Benchmark: average over BENCH_RUNS ───────────────────────────────────
    printf("\nBenchmarking (%d runs)...\n", BENCH_RUNS);
    gettimeofday(&t0, NULL);
    for (int i = 0; i < BENCH_RUNS; ++i) {
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        rknn_outputs_release(ctx, io_num.n_output, outputs);
    }
    gettimeofday(&t1, NULL);
    double avg_ms = (get_ms(t1) - get_ms(t0)) / BENCH_RUNS;
    printf("[Average inference time] %.2f ms  (%.1f FPS)\n", avg_ms, 1000.0 / avg_ms);

    rknn_destroy(ctx);
    return 0;
}
