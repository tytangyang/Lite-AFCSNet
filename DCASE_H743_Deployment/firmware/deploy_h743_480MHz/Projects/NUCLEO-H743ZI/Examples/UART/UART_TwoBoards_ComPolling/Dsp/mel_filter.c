/**
 * @file    mel_filter.c
 * @brief   Mel 滤波器组实现 —— 运行时计算，稀疏存储
 *
 * 三角滤波：每个 Mel 滤波器在 Mel 刻度上是中心对称的三角窗，
 * 映射回 Hz 再离散化到 FFT bin 后只覆盖少量 bin。
 * 采用稀疏存储避免 256×2049 的密集矩阵。
 */

#include "mel_filter.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ---------- 静态内存池 ---------- */
#define MAX_SPARSE_WEIGHTS  6000   /* 256 个滤波器 × 平均 ~20 bin */

static mel_filter_t  s_filters[MEL_N_MELS];       /* 256 个滤波器头 */
static float         s_weights[MAX_SPARSE_WEIGHTS]; /* 所有权重串联存储 */
static uint16_t      s_total_weights = 0;           /* 实际使用的权重数 */

/* ---------- Mel 刻度转换 ---------- */

/* 线性频率 -> Mel 频率 */
static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

/* Mel 频率 -> 线性频率 */
static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* ---------- 初始化 ---------- */

int mel_filter_init(void) {
    float mel_pts[MEL_N_MELS + 2];   /* 256 个中心 + 2 个边界 */
    float f_pts[MEL_N_MELS + 2];     /* 对应的 Hz */
    float bin_pts[MEL_N_MELS + 2];   /* 对应的 FFT bin (浮点) */
    int   left_bin[MEL_N_MELS + 2];  /* 向下取整 */
    int   m;

    /* 1. 计算 Mel 刻度上的等距分点 (n_mels+2 个) */
    float mel_min = hz_to_mel(MEL_F_MIN);
    float mel_max = hz_to_mel(MEL_F_MAX);
    float mel_step = (mel_max - mel_min) / (MEL_N_MELS + 1);

    for (m = 0; m < MEL_N_MELS + 2; m++) {
        mel_pts[m] = mel_min + m * mel_step;
        f_pts[m]   = mel_to_hz(mel_pts[m]);
        bin_pts[m] = f_pts[m] * MEL_N_FFT / MEL_SAMPLE_RATE;
        left_bin[m]  = (int)floorf(bin_pts[m]);
    }

    /* 2. 为每个 Mel 滤波器生成稀疏权重 */
    s_total_weights = 0;

    for (m = 0; m < MEL_N_MELS; m++) {
        /* 滤波器 m 的左/中/右 位置是 mel_pts[m], mel_pts[m+1], mel_pts[m+2] */
        int start  = left_bin[m];       /* 左边界取整 */
        int end    = left_bin[m + 2];   /* 右边界取整 */

        /* 注意: 峰值位置用分数 bin_pts[m+1] 判断上升/下降分支 (与 torchaudio
         * 一致)。不能用 roundf 取整 —— 低频滤波器只有几个 bin 宽, 峰值取整
         * 会形成"平顶", 与参考滤波器组偏差可达 0.4 */
        float peak = bin_pts[m + 1];

        /* 限定在有效 bin 范围内 [0, MEL_N_BINS-1] */
        if (start < 0) start = 0;
        if (end >= MEL_N_BINS) end = MEL_N_BINS - 1;
        if (end < start) end = start;

        int num_bins = end - start + 1;
        if (num_bins <= 0) {
            /* 极端边缘情况：滤波器范围为空，跳过一个 bin */
            start = (m == 0) ? 0 : left_bin[m] - 1;
            end = start + 1;
            if (end >= MEL_N_BINS) end = MEL_N_BINS - 1;
            num_bins = end - start + 1;
        }

        if (s_total_weights + num_bins > MAX_SPARSE_WEIGHTS) {
            printf("[MEL] ERR: sparse weights overflow (need %d, max %d)\r\n",
                   s_total_weights + num_bins, MAX_SPARSE_WEIGHTS);
            return -1;
        }

        /* 记录滤波器头 */
        s_filters[m].start_bin = (uint16_t)start;
        s_filters[m].num_bins  = (uint16_t)num_bins;
        s_filters[m].weights   = &s_weights[s_total_weights];

        /* 计算三角权重 */
        for (int bin = start; bin <= end; bin++) {
            int idx = bin - start;
            float weight;

            if ((float)bin <= peak) {
                /* 上升沿: bin ∈ [start, peak] */
                float denom = bin_pts[m + 1] - bin_pts[m];   /* peak - left */
                if (denom < 1e-6f) denom = 1e-6f;
                weight = (bin - bin_pts[m]) / denom;
            } else {
                /* 下降沿: bin ∈ (peak, end] */
                float denom = bin_pts[m + 2] - bin_pts[m + 1]; /* right - peak */
                if (denom < 1e-6f) denom = 1e-6f;
                weight = (bin_pts[m + 2] - bin) / denom;
            }

            /* 裁到 [0, 1] */
            if (weight < 0.0f) weight = 0.0f;
            if (weight > 1.0f) weight = 1.0f;

            s_weights[s_total_weights + idx] = weight;
        }

        s_total_weights += num_bins;
    }

    return 0;
}

/* ---------- 获取滤波器 ---------- */

const mel_filter_t* mel_filter_get(int mel_idx) {
    if (mel_idx < 0 || mel_idx >= MEL_N_MELS) return NULL;
    return &s_filters[mel_idx];
}

/* ---------- 调试 ---------- */

void mel_filter_print_stats(void) {
    printf("[MEL] %d filters, %d non-zero weights (~%.1f KB)\r\n",
           MEL_N_MELS, s_total_weights,
           s_total_weights * sizeof(float) / 1024.0f);

    for (int m = 0; m < MEL_N_MELS; m += 32) {
        printf("[MEL] filter %3d: bin [%4d, %4d] %3d weights\r\n",
               m, s_filters[m].start_bin,
               s_filters[m].start_bin + s_filters[m].num_bins - 1,
               s_filters[m].num_bins);
    }
}
