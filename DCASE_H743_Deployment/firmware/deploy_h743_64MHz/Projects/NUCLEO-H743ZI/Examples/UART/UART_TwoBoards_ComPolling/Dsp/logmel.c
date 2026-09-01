/**
 * @file    logmel.c
 * @brief   Log-Mel 频谱图计算实现
 *
 * 内部流程 (每帧)：
 *   1. 取 3072 采样点 + Hann 窗
 *   2. 补零到 4096 → arm_rfft_fast_f32
 *   3. 计算功率谱 |FFT|^2 (2049 bins)
 *   4. 稀疏 Mel 滤波 (256 filters × sparse weights)
 *   5. ln(energy + 1e-5)
 *
 * 内存 (静态)：
 *   FFT instance  ~8 KB  (CMSIS-DSP 内部分配)
 *   Hann window   ~12 KB (3072 floats)
 *   FFT buffer    ~16 KB (4096 floats, 原地计算)
 *   总计 ~36 KB SRAM
 */

#include "logmel.h"
#include "mel_filter.h"
#include "arm_math.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ---------- 静态缓冲区 ---------- */

static arm_rfft_fast_instance_f32  s_fft_inst;        /* FFT 实例 */
static float                        s_hann[LOGMEL_WIN_LENGTH]; /* Hann 窗 */
static float                        s_fft_buf[LOGMEL_N_FFT];   /* FFT 输入 */
static float                        s_fft_out[LOGMEL_N_FFT];   /* FFT 输出 (与输入分离!) */
static float                        s_fft_cap[LOGMEL_N_FFT];   /* 调试: 第 0 帧 FFT 输出快照 */
static int                          s_initialized = 0;

/* ---------- 初始化 ---------- */

int logmel_init(void) {
    if (s_initialized) return 0;

    /* 1. 初始化 FFT (4096 点) */
    arm_status status = arm_rfft_fast_init_f32(&s_fft_inst, LOGMEL_N_FFT);
    if (status != ARM_MATH_SUCCESS) {
        printf("[LOGMEL] ERR: FFT init failed\r\n");
        return -1;
    }

    /* 2. 生成 Hann 窗 (periodic=True, 与 torch.hann_window 对齐) */
    int win_len = LOGMEL_WIN_LENGTH;
    for (int i = 0; i < win_len; i++) {
        s_hann[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * i / win_len));
    }

    /* 3. 初始化 Mel 滤波器组 */
    if (mel_filter_init() != 0) {
        printf("[LOGMEL] ERR: mel filter init failed\r\n");
        return -2;
    }

    s_initialized = 1;
    printf("[LOGMEL] init OK (FFT %d, win %d, hop %d, mel %d)\r\n",
           LOGMEL_N_FFT, LOGMEL_WIN_LENGTH, LOGMEL_HOP_LENGTH, LOGMEL_N_MELS);
    return 0;
}

/* ---------- 核心计算 ---------- */

int logmel_compute(const int16_t *pcm, int pcm_len, float *logmel_out) {
    if (!s_initialized) return -1;
    if (pcm == NULL || logmel_out == NULL) return -1;
    if (pcm_len < LOGMEL_PCM_LEN) return -1;

    int frame, mel_idx, i;

    for (frame = 0; frame < LOGMEL_N_FRAMES; frame++) {
        /* --- 1. 取当前帧的 3072 个采样 + Hann 窗 ---
         * torch.stft(center=True, pad_mode="reflect"): 帧 t 的窗覆盖原始样本
         * [t*hop - n_fft/2, t*hop - n_fft/2 + win_length)
         * 越界采样用 reflect 填充 (src<0 -> -src, src>=N -> 2N-1-src)
         * 注意: 若直接取 [t*hop, t*hop+win) 会整体右移 2048 点 (64ms),
         * 特征与训练 pipeline 时间对齐错位 */
        int start = frame * LOGMEL_HOP_LENGTH - LOGMEL_N_FFT / 2;

        for (i = 0; i < LOGMEL_WIN_LENGTH; i++) {
            int src = start + i;
            if (src < 0) src = -src;
            else if (src >= pcm_len) src = 2 * pcm_len - 2 - src;
            /* int16 PCM 归一化到 [-1,1] (对齐训练 pipeline torchaudio.load) */
            s_fft_buf[i] = (float)pcm[src] * (1.0f / 32768.0f) * s_hann[i];
        }
        /* 补零到 4096 */
        for (i = LOGMEL_WIN_LENGTH; i < LOGMEL_N_FFT; i++) {
            s_fft_buf[i] = 0.0f;
        }

        /* --- 2. FFT (输入/输出分离, 不能原地!) ---
         * arm_rfft_fast_f32 的 stage_rfft 阶段边读输入边写输出,
         * 若 pOut==p, 高频 bin (尾部) 读到的输入已被覆盖 → 12-16kHz 能量爆炸 */
        arm_rfft_fast_f32(&s_fft_inst, s_fft_buf, s_fft_out, 0);

        if (frame == 0) {
            memcpy(s_fft_cap, s_fft_out, sizeof(s_fft_cap)); /* 调试快照 */
        }

        /* --- 3. 功率谱 & Mel 滤波 --- */
        /* CMSIS-DSP RFFT 输出格式 (N=4096):
         *   buf[0]   = Re[0]    (DC)
         *   buf[1]   = Re[N/2]  (Nyquist)
         *   buf[2k]   = Re[k],  buf[2k+1] = Im[k]   for k=1..N/2-1
         */

        float mel_energy[LOGMEL_N_MELS];

        for (mel_idx = 0; mel_idx < LOGMEL_N_MELS; mel_idx++) {
            const mel_filter_t *mf = mel_filter_get(mel_idx);
            if (mf == NULL) {
                mel_energy[mel_idx] = 0.0f;
                continue;
            }

            float sum = 0.0f;

            for (int bin = 0; bin < mf->num_bins; bin++) {
                int k = mf->start_bin + bin;
                float re, im, power;

                if (k == 0) {
                    /* DC */
                    re = s_fft_out[0];
                    power = re * re;
                } else if (k == LOGMEL_N_FFT / 2) {
                    /* Nyquist */
                    re = s_fft_out[1];
                    power = re * re;
                } else {
                    /* k in [1, N/2-1] */
                    int idx = 2 * k;
                    re = s_fft_out[idx];
                    im = s_fft_out[idx + 1];
                    power = re * re + im * im;
                }

                sum += power * mf->weights[bin];
            }

            mel_energy[mel_idx] = sum;
        }

        /* --- 4. Log 变换 (自然对数, ln) --- */
        /* 对应 Python: (mel + 1e-5).log() */
        /* 输出布局: [mel_bins, frames] = output[mel * n_frames + frame] */
        /* 匹配模型输入 [1, 1, 256, 65] */
        for (mel_idx = 0; mel_idx < LOGMEL_N_MELS; mel_idx++) {
            logmel_out[mel_idx * LOGMEL_N_FRAMES + frame] =
                logf(mel_energy[mel_idx] + 1e-5f);
        }
    }

    return 0;
}

/* ---------- 调试 ---------- */

float *logmel_get_fft_cap(void) { return s_fft_cap; }

void logmel_print_meminfo(void) {
    printf("[LOGMEL] FFT buf:  %d B\r\n", (int)(LOGMEL_N_FFT * sizeof(float)));
    printf("[LOGMEL] Hann win: %d B\r\n", (int)(LOGMEL_WIN_LENGTH * sizeof(float)));
    printf("[LOGMEL] Total:    ~%d KB\r\n",
           (int)((LOGMEL_N_FFT * sizeof(float) +
                  LOGMEL_WIN_LENGTH * sizeof(float) +
                  sizeof(s_fft_inst)) / 1024));
}
