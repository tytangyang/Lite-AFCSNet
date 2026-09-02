/**
 * @file    logmel.h
 * @brief   Log-Mel 频谱图计算
 *
 * 输入：int16 PCM 原始音频 (32kHz, 35072 samples)
 * 输出：float [256 mel × 65 frames]，mel-major 排列
 *       logmel_out[mel_idx * 65 + frame_idx]
 *       匹配模型输入 [1, 1, 256, 65]
 *
 * 对应 Python 端：
 *   torchaudio.transforms.MelSpectrogram(sr=32000, n_fft=4096,
 *       win_length=3072, hop_length=500, n_mels=256)
 *   -> (spectrogram + 1e-5).log()
 */

#ifndef LOGMEL_H
#define LOGMEL_H

#include <stdint.h>

#define LOGMEL_SAMPLE_RATE   32000
#define LOGMEL_N_FFT         4096
#define LOGMEL_WIN_LENGTH    3072
#define LOGMEL_HOP_LENGTH    500
#define LOGMEL_N_MELS        256
#define LOGMEL_N_FRAMES      65

/* 一帧 PCM 音频的采样点数 */
#define LOGMEL_PCM_LEN   (LOGMEL_WIN_LENGTH + (LOGMEL_N_FRAMES - 1) * LOGMEL_HOP_LENGTH)
/* 35072 samples @ 32kHz ≈ 1.10 秒 */

/* 输出张量大小 [256 mel, 65 frames] */
#define LOGMEL_OUT_SIZE  (LOGMEL_N_MELS * LOGMEL_N_FRAMES)

/**
 * @brief  初始化 log-mel 模块 (上电后调用一次)
 *         内部完成：FFT 初始化、Hann 窗生成、Mel 滤波器初始化
 * @return 0 成功, 非0 失败
 */
int logmel_init(void);

/**
 * @brief  从 PCM 原始数据计算 log-mel 频谱图
 * @param  pcm      输入 PCM 采样 (int16, LOGMEL_PCM_LEN 个)
 * @param  pcm_len  输入长度
 * @param  logmel_out 输出 float[256*65] row-major
 *                   mel_out[mel_idx * 65 + frame_idx]
 * @return 0 成功, -1 参数错误
 */
int logmel_compute(const int16_t *pcm, int pcm_len, float *logmel_out);

/**
 * @brief  打印内存使用统计 (调试用)
 */
void logmel_print_meminfo(void);

/**
 * @brief  获取第 0 帧 FFT 输出快照 (调试用, LOGMEL_N_FFT 个 float)
 */
float *logmel_get_fft_cap(void);

#endif /* LOGMEL_H */
