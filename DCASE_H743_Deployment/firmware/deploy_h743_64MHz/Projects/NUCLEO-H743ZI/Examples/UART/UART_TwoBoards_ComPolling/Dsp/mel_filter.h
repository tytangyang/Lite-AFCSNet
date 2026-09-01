/**
 * @file    mel_filter.h
 * @brief   Mel 滤波器组 —— 稀疏存储，运行时初始化
 *
 * 参数 (对应 DCASE 训练端)：
 *   sample_rate = 32000, n_fft = 4096, n_mels = 256
 *   f_min = 0, f_max = 16000
 *
 * 稀疏表示：
 *   256 个三角滤波器，每个在 FFT bins (0..2048) 上只覆盖约 8~20 个非零权重。
 *   总共约 4000 个 float 权重，~16 KB，远小于密集矩阵 256×2049≈2MB。
 */

#ifndef MEL_FILTER_H
#define MEL_FILTER_H

#include <stdint.h>

/* DSP 参数 (固定，与训练端对齐) */
#define MEL_SAMPLE_RATE    32000
#define MEL_N_FFT          4096
#define MEL_N_MELS         256
#define MEL_F_MIN          0.0f
#define MEL_F_MAX          16000.0f

/* FFT 输出的频点数 (n_fft/2 + 1) */
#define MEL_N_BINS         (MEL_N_FFT / 2 + 1)  /* 2049 */

/* ---------- 稀疏滤波器结构 ---------- */
typedef struct {
    uint16_t  start_bin;      /* 第一个非零的 FFT bin 下标 */
    uint16_t  num_bins;       /* 非零权重的数量 */
    const float *weights;     /* 指向权重数组的指针 (num_bins 项) */
} mel_filter_t;

/**
 * @brief  初始化 Mel 滤波器组 (上电后调用一次)
 * @return 0 成功, -1 失败 (内存不足)
 */
int mel_filter_init(void);

/**
 * @brief  获取第 mel_idx 个滤波器 (0 ~ 255)
 */
const mel_filter_t* mel_filter_get(int mel_idx);

/**
 * @brief  打印滤波器统计信息 (调试用)
 */
void mel_filter_print_stats(void);

#endif /* MEL_FILTER_H */
