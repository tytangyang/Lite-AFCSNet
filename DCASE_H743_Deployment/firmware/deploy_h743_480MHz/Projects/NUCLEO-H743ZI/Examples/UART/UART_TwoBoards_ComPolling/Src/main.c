/**
 * @file    main.c
 * @brief   DCASE 音频场景分类部署 —— 基于官方 UART_TwoBoards_ComPolling 例程移植
 *
 * 链路: PC(串口脚本) → USART3 PD8/PD9 @115200 (ST-Link VCP)
 *     → PCM 原始音频 (35072×int16 = 70144 B)
 *     → Log-Mel DSP → [256×65] float
 *     → int8 量化 → X-CUBE-AI 推理 (BC-ResNet int8)
 *     → 10 类结果 → USART3 回传 PC 显示
 *
 * 基座来自官方例程（已验证可运行）:
 *   - CPU_CACHE_Enable + MPU_Config (D2 SRAM 标记 non-cacheable)
 *   - SystemClock_Config: 480MHz 超频版 (LDO + VOS0 + HSE 25MHz PLL, 见下文)
 *   - USART3 轮询收发 @ 115200 -> ST-Link VCP
 *
 * 协议 (板端自同步):
 *   板端循环等待 4 字节魔数 0xA5 0x5A 0xA5 0x5A, 随后分块接收 70144 B PCM;
 *   处理完成后回传一行结果:
 *     RESULT class=<id>,<name>,conf=<float>,dsp=<ms>,inf=<ms>,total=<ms>
 *
 * 说明:
 *   - HAL_UART_Receive 单次 Size 为 uint16 (上限 65535), 70KB 需分块轮询接收
 *   - pcm_frame 放 D2 SRAM (scatter: .PcmFrameD2), AI activation 等放 AXI SRAM
 *   - 栈在 startup 文件中已加大到 16KB (logmel_compute 局部 mel_energy[256]=1KB)
 */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "logmel.h"
#include "mel_filter.h"
#include "network.h"
#include "network_data.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ================================================================
 * 配置项
 * ================================================================ */
#define UART_BAUDRATE           115200

/* 推理耗时测量引脚 (PA0 -> 示波器/逻辑分析仪) */
#define PROFILE_PORT            GPIOA
#define PROFILE_PIN             GPIO_PIN_0

/* 量化参数 (来自 X-CUBE-AI generate report) */
#define INPUT_SCALE      0.094224639f
#define INPUT_ZERO_POINT (-8)
#define OUTPUT_SCALE     0.046166364f
#define OUTPUT_ZERO_POINT (-33)

/* 帧接收魔数 (PC 端发送脚本同步用) */
static const uint8_t PCM_MAGIC[4]  = {0xA5, 0x5A, 0xA5, 0x5A};
static const uint8_t DUMP_MAGIC[4] = {0x5A, 0xA5, 0x5A, 0xA5}; /* 调试: 回传 logmel */

/* 接收分块大小: < 65535, 且为偶数 (int16 对齐) */
#define RX_CHUNK_SIZE           4096

/* ================================================================
 * 全局句柄
 * ================================================================ */
UART_HandleTypeDef huart3;

/* PCM 缓冲: 放 D2 SRAM (scatter 文件 .PcmFrameD2),
 * 与 AXI SRAM 中的 AI/DSP 大数据分离, 预留 AXI 空间给 activation */
__attribute__((section(".PcmFrameD2")))
__attribute__((aligned(4)))
int16_t pcm_frame[LOGMEL_PCM_LEN];

/* ================================================================
 * AI 推理资源 (AXI SRAM, .ANY 放置)
 * ================================================================ */
static AI_ALIGNED(4)
uint8_t s_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];   /* 319872 B */

static int8_t  s_ai_input[AI_NETWORK_IN_1_SIZE];           /* 16640 int8 */
static int8_t  s_ai_output_raw[AI_NETWORK_OUT_1_SIZE];     /* 10 int8 */
static float   s_ai_output_float[AI_NETWORK_OUT_1_SIZE];   /* 反量化后 */

/* log-mel 结果缓冲: 放 D2 SRAM (AXI 512KB 已被 activation 319KB 等占用,
 * D2 288KB 中 PCM 仅占 70KB, 剩余空间充足) */
__attribute__((section(".DspBufD2")))
__attribute__((aligned(4)))
static float   s_logmel[LOGMEL_OUT_SIZE];                  /* log-mel float */

static ai_handle s_network = AI_HANDLE_NULL;

/* ================================================================
 * 函数声明
 * ================================================================ */
static void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void dwt_init(void);
static int  ai_init(void);
static int  ai_run(const float *logmel, float *output);
static int  uart_wait_magic(void);
static int  uart_recv_pcm(void);
static int  argmax(const float *probs, int n);
static void Error_Handler(void);
static void CPU_CACHE_Enable(void);

/* ================================================================
 * UART 接收 (分块轮询, 自同步)
 * ================================================================ */

/* 等待 4 字节魔数, 支持字节级自同步
 * 返回: 0 = 正常推理模式 (PCM_MAGIC), 1 = 调试 dump 模式 (DUMP_MAGIC) */
static int uart_wait_magic(void)
{
    uint8_t b;
    int idx = 0;
    int dump = 0;

    for (;;) {
        if (HAL_UART_Receive(&huart3, &b, 1, 1000) != HAL_OK) {
            idx = 0;                  /* 超时: 重新同步 */
            continue;
        }
        /* 同时跟踪两套魔数 (按当前匹配位置对齐比较) */
        const uint8_t *m = dump ? DUMP_MAGIC : PCM_MAGIC;
        if (b == m[idx]) {
            idx++;
            if (idx == 4) return dump;
        } else {
            /* 尝试另一套魔数重新对齐 */
            if (b == PCM_MAGIC[0])      { idx = 1; dump = 0; }
            else if (b == DUMP_MAGIC[0]) { idx = 1; dump = 1; }
            else                         { idx = 0; }
        }
    }
}

/* 接收完整 PCM 帧 (70144 B), 返回 0 成功
 *
 * 注意: 本函数执行期间不得有任何阻塞 TX (printf 等)——magic 之后的
 * PCM 流是连续的, 任何 >86µs 的 TX 阻塞都会使 1 字节深的 RDR 过载
 * (ORE), 导致 HAL_UART_Receive 返回 HAL_ERROR。调用方必须确保
 * magic 识别到 PCM 接收完成之间不发送任何字符。
 */
static int uart_recv_pcm(void)
{
    uint8_t *p  = (uint8_t *)pcm_frame;
    uint32_t total = LOGMEL_PCM_LEN * 2u;
    uint32_t done  = 0;

    while (done < total) {
        uint32_t chunk = (total - done > RX_CHUNK_SIZE) ? RX_CHUNK_SIZE : (total - done);
        /* 每块 4096 B @115200 ≈ 356ms, 5s 超时 (容忍 ST-Link VCP 短暂停滞) */
        HAL_StatusTypeDef st = HAL_UART_Receive(&huart3, p + done, (uint16_t)chunk, 5000);
        if (st != HAL_OK) {
            printf("[UART] chunk@%lu/%lu fail st=%d err=0x%08lx ISR=0x%08lx\r\n",
                   (unsigned long)done, (unsigned long)total, (int)st,
                   (unsigned long)huart3.ErrorCode,
                   (unsigned long)huart3.Instance->ISR);
            return -1;
        }
        done += chunk;
    }
    return 0;
}

/* ================================================================
 * DWT 周期计数器 (耗时统计)
 * ================================================================ */
static void dwt_init(void)
{
    /* Cortex-M7: DWT 带软件锁 (DWT_LSR.SLK), 上锁时对 DWT 的写入被忽略,
     * 需先写 DWT_LAR 解锁钥匙 (参考 uC/OS-III bsp_cpu.c) */
    if ((DWT->LSR & (1u << 1u)) != 0u) {   /* SLK = 1 已上锁 */
        DWT->LAR = 0xC5ACCE55u;            /* unlock key */
    }
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline uint32_t dwt_get_cycles(void) { return DWT->CYCCNT; }

static inline float cycles_to_ms(uint32_t c)
{
    return (float)c / (SystemCoreClock / 1000.0f);
}

/* ================================================================
 * AI 推理
 * ================================================================ */
static int ai_init(void)
{
    ai_error err;
    /* X-CUBE-AI 10.2 新 API: 传入 activation / weights 缓冲区句柄数组 */
    const ai_handle acts[] = { (ai_handle)s_activations };
    const ai_handle wts[]  = { (ai_handle)s_network_weights_array_u64 };

    err = ai_network_create_and_init(&s_network, acts, wts);
    if (err.type != AI_ERROR_NONE) {
        printf("[AI] init failed: type=%d, code=%d\r\n", err.type, err.code);
        return -1;
    }
    printf("[AI] network ready (activations=%d B, weights=%d B)\r\n",
           AI_NETWORK_DATA_ACTIVATIONS_SIZE, (int)sizeof(s_network_weights_array_u64));
    return 0;
}

/* float → int8 量化 */
static void quantize_input(const float *src, int8_t *dst, int n)
{
    for (int i = 0; i < n; i++) {
        float v = src[i] / INPUT_SCALE + (float)INPUT_ZERO_POINT;
        int32_t q = (int32_t)roundf(v);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        dst[i] = (int8_t)q;
    }
}

/* int8 → float 反量化 */
static void dequantize_output(const int8_t *src, float *dst, int n)
{
    for (int i = 0; i < n; i++) {
        dst[i] = OUTPUT_SCALE * ((float)src[i] - (float)OUTPUT_ZERO_POINT);
    }
}

/* logits → 概率 (数值稳定性: 减去最大值) */
static void softmax(const float *in, float *out, int n)
{
    float m = in[0];
    for (int i = 1; i < n; i++)
        if (in[i] > m) m = in[i];

    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(in[i] - m);
        s += out[i];
    }
    for (int i = 0; i < n; i++) out[i] /= s;
}

static int ai_run(const float *logmel, float *output)
{
    ai_buffer *ai_in  = ai_network_inputs_get(s_network, NULL);
    ai_buffer *ai_out = ai_network_outputs_get(s_network, NULL);
    ai_i32 batch;

    /* 1. float logmel → int8 */
    quantize_input(logmel, s_ai_input, AI_NETWORK_IN_1_SIZE);

    /* 2. 绑定输入/输出 buffer */
    ai_in->data  = AI_HANDLE_PTR(s_ai_input);
    ai_out->data = AI_HANDLE_PTR(s_ai_output_raw);

    /* 3. 推理 */
    batch = ai_network_run(s_network, ai_in, ai_out);
    if (batch != 1) {
        ai_error err = ai_network_get_error(s_network);
        printf("[AI] run failed: batch=%d, err=%d,%d\r\n",
               (int)batch, err.type, err.code);
        return -1;
    }

    /* 4. int8 → float → softmax 概率 */
    dequantize_output(s_ai_output_raw, output, AI_NETWORK_OUT_1_SIZE);
    softmax(output, output, AI_NETWORK_OUT_1_SIZE);
    return 0;
}

/* ================================================================
 * 分类结果
 * ================================================================ */
static const char *CLASS_NAMES[10] = {
    "airport", "bus", "metro", "metro_station", "park",
    "public_square", "shopping_mall", "street_pedestrian",
    "street_traffic", "tram"
};

static int argmax(const float *probs, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (probs[i] > probs[best]) best = i;
    return best;
}

/* ================================================================
 * 主程序
 * ================================================================ */
int main(void)
{
    /* 1. 使能 Cache + MPU (D2 SRAM non-cacheable) */
    CPU_CACHE_Enable();
    MPU_Config();

    /* 2. HAL 初始化 + 时钟 (480MHz, 见 SystemClock_Config) */
    HAL_Init();
    SystemClock_Config();

    /* 3. LED 与测量引脚 */
    BSP_LED_Init(LED1);
    BSP_LED_Init(LED2);
    BSP_LED_Init(LED3);
    MX_GPIO_Init();

    /* 4. USART3 (PD8/PD9, ST-Link VCP) 轮询模式 */
    huart3.Instance        = USART3;
    huart3.Init.BaudRate   = UART_BAUDRATE;
    huart3.Init.WordLength = UART_WORDLENGTH_8B;
    huart3.Init.StopBits   = UART_STOPBITS_1;
    huart3.Init.Parity     = UART_PARITY_NONE;
    huart3.Init.HwFlowCtl  = UART_HWCONTROL_NONE;
    huart3.Init.Mode       = UART_MODE_TX_RX;
    huart3.Init.OverSampling = UART_OVERSAMPLING_16;
    huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;

    if (HAL_UART_Init(&huart3) != HAL_OK) {
        Error_Handler();
    }

    dwt_init();

    printf("\r\n=== DCASE Scene Classifier (deploy_h743_480MHz) ===\r\n");
    printf("SysClk: %lu Hz, UART: %ld baud (ST-Link VCP)\r\n",
           SystemCoreClock, huart3.Init.BaudRate);

    /* 5. DSP 初始化 */
    if (logmel_init() != 0) {
        printf("[FATAL] DSP init failed!\r\n");
        Error_Handler();
    }
    logmel_print_meminfo();
    mel_filter_print_stats();

    /* 6. AI 初始化 */
    if (ai_init() != 0) {
        printf("[FATAL] AI init failed!\r\n");
        Error_Handler();
    }

    /* 7. 主循环: 等音频 → DSP → AI → 回传结果 */
    while (1)
    {
        uint32_t t_total, t_dsp, t_inf;

        BSP_LED_On(LED1);
        printf("\r\n[UART] waiting for PCM (%d samples = %u B)...\r\n",
               LOGMEL_PCM_LEN, (unsigned)(LOGMEL_PCM_LEN * 2));

        if (uart_wait_magic() != 0) {
            /* ====== 调试 dump 模式: 计算 logmel 后回传原始 float32 ====== */
            if (uart_recv_pcm() != 0) {
                printf("[UART] dump RX timeout\r\n");
                continue;
            }
            logmel_compute(pcm_frame, LOGMEL_PCM_LEN, s_logmel);
            /* 先发 FFT 快照 (4096×4=16KB), 再发 logmel (16640×4=66560B),
             * 最后发反量化 logits (10×4=40B) —— 均带 4 字节长度头;
             * HAL_UART_Transmit 的 Size 是 uint16, 需分块 */
            ai_run(s_logmel, s_ai_output_float);
            const uint8_t *parts[3];
            uint32_t part_lens[3];
            parts[0]     = (const uint8_t *)logmel_get_fft_cap();
            parts[1]     = (const uint8_t *)s_logmel;
            parts[2]     = (const uint8_t *)s_ai_output_float;
            part_lens[0] = LOGMEL_N_FFT * sizeof(float);
            part_lens[1] = LOGMEL_OUT_SIZE * sizeof(float);
            part_lens[2] = AI_NETWORK_OUT_1_SIZE * sizeof(float);
            for (int part = 0; part < 3; part++) {
                const uint32_t plen = part_lens[part];
                const uint8_t *pp   = parts[part];
                HAL_UART_Transmit(&huart3, (uint8_t *)&plen, 4, 2000);
                uint32_t sent = 0;
                while (sent < plen) {
                    uint16_t n = (plen - sent > 4096) ? 4096 : (uint16_t)(plen - sent);
                    HAL_UART_Transmit(&huart3, (uint8_t *)(pp + sent), n, 5000);
                    sent += n;
                }
            }
            printf("[DUMP] sent FFT(%uB) + logmel(%uB) + logits(%uB)\r\n",
                   (unsigned)part_lens[0], (unsigned)part_lens[1],
                   (unsigned)part_lens[2]);
            continue;
        }
        /* 注意: 这里禁止 printf —— magic 后的 PCM 流连续到达,
         * 任何阻塞 TX 都会造成 RDR 过载 (ORE), 详见 uart_recv_pcm() */

        if (uart_recv_pcm() != 0) {
            printf("[UART] RX timeout, please send again\r\n");
            continue;
        }
        BSP_LED_Off(LED1);

        /* DSP + AI 处理 */
        /* DWT 计数器会被调试器 detach 清掉 (TRCENA=0 时读回全 0),
         * 每次测量前强制重新使能计数 */
        dwt_init();

        t_total = dwt_get_cycles();

        t_dsp = dwt_get_cycles();
        logmel_compute(pcm_frame, LOGMEL_PCM_LEN, s_logmel);
        t_dsp = dwt_get_cycles() - t_dsp;

        HAL_GPIO_WritePin(PROFILE_PORT, PROFILE_PIN, GPIO_PIN_SET);
        t_inf = dwt_get_cycles();
        ai_run(s_logmel, s_ai_output_float);
        t_inf = dwt_get_cycles() - t_inf;
        HAL_GPIO_WritePin(PROFILE_PORT, PROFILE_PIN, GPIO_PIN_RESET);

        t_total = dwt_get_cycles() - t_total;

        /* 结果回传: class + conf + 完整 10 类概率 (供 PC 对照 int8 差异) */
        int cls = argmax(s_ai_output_float, AI_NETWORK_OUT_1_SIZE);

        char buf[192];
        int len = snprintf(buf, sizeof(buf),
            "RESULT class=%d,%s,conf=%.3f,dsp=%.1fms(%lu),inf=%.1fms(%lu),total=%.1fms(%lu)\r\n",
            cls, CLASS_NAMES[cls], s_ai_output_float[cls],
            cycles_to_ms(t_dsp), (unsigned long)t_dsp,
            cycles_to_ms(t_inf), (unsigned long)t_inf,
            cycles_to_ms(t_total), (unsigned long)t_total);
        HAL_UART_Transmit(&huart3, (uint8_t *)buf, len, 1000);
        printf("%s", buf);

        /* 第二行: probs=p0,p1,...,p9 (softmax 后, 逗号分隔) */
        int plen = snprintf(buf, sizeof(buf), "PROBS=");
        for (int k = 0; k < AI_NETWORK_OUT_1_SIZE; k++) {
            int n = snprintf(buf + plen, sizeof(buf) - (size_t)plen,
                             "%.4f%s", s_ai_output_float[k],
                             (k < AI_NETWORK_OUT_1_SIZE - 1) ? "," : "\r\n");
            plen += n;
        }
        HAL_UART_Transmit(&huart3, (uint8_t *)buf, (uint16_t)plen, 1000);

        BSP_LED_Toggle(LED2);
    }
}

/* ================================================================
 * printf 重定向 → USART3 (完整 C 库, 禁用半主机)
 *
 * 背景: 完整 C 库启动时用 freopen(":tt") 初始化 stdout, 内部调用
 * 半主机 _sys_open() -> BKPT #0xab。裸机无调试器时 BKPT 直接
 * HardFault, 表现为"上电后串口无输出 / 卡死"。
 * 修复: 自定义 __stdout/__stdin + __use_no_semihosting, 让链接器
 * 跳过库的流初始化路径, printf 直接走下面的 fputc。
 * ================================================================ */
#pragma import(__use_no_semihosting)

struct __FILE { int handle; };
FILE __stdout;
FILE __stdin;

int fputc(int ch, FILE *f)
{
    (void)f;
    while (!(huart3.Instance->ISR & USART_ISR_TXE_TXFNF));
    huart3.Instance->TDR = (uint8_t)ch;
    return ch;
}

int fgetc(FILE *f)
{
    (void)f;
    return EOF;   /* 不使用 scanf, 恒返回 EOF */
}

int ferror(FILE *f)
{
    (void)f;
    return EOF;
}

void _sys_exit(int return_code)
{
    (void)return_code;
    for (;;) { }
}

void _ttywrch(int ch)
{
    (void)ch;
}

/* ================================================================
 * MPU 配置: D2 SRAM (0x30000000, 256KB) non-cacheable
 * pcm_frame 位于 D2 SRAM, 保证后续改用 DMA 时无 cache 一致性问题
 * ================================================================ */
static void MPU_Config(void)
{
    MPU_Region_InitTypeDef MPU_InitStruct = {0};

    HAL_MPU_Disable();

    /* D2 SRAM: DMA/CPU 共享缓冲 → 禁用 cache */
    MPU_InitStruct.Enable           = MPU_REGION_ENABLE;
    MPU_InitStruct.BaseAddress      = 0x30000000;
    MPU_InitStruct.Size             = MPU_REGION_SIZE_256KB;
    MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;
    MPU_InitStruct.IsBufferable     = MPU_ACCESS_NOT_BUFFERABLE;
    MPU_InitStruct.IsCacheable      = MPU_ACCESS_NOT_CACHEABLE;
    MPU_InitStruct.IsShareable      = MPU_ACCESS_SHAREABLE;
    MPU_InitStruct.Number           = MPU_REGION_NUMBER0;
    MPU_InitStruct.TypeExtField     = MPU_TEX_LEVEL0;
    MPU_InitStruct.SubRegionDisable = 0x00;
    MPU_InitStruct.DisableExec      = MPU_INSTRUCTION_ACCESS_DISABLE;

    HAL_MPU_ConfigRegion(&MPU_InitStruct);

    /* 使能 MPU, 未配置区域使用默认内存映射 (AXI SRAM 保持 cacheable) */
    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

/* ================================================================
 * System Clock Configuration —— 480MHz 版 (deploy_h743_480MHz 副本)
 *
 *   25MHz HSE / PLLM(5) = 5MHz ref
 *   VCO = 5MHz * PLLN(192) = 960MHz (宽范围 RCC_PLL1VCOWIDE)
 *   SYSCLK = VCO / PLLP(2) = 480MHz
 *
 *   关键前提 (480MHz 必须, 否则起不来):
 *   1) HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY) 供电路径
 *   2) VOS0 电压档 + 等待 VOSRDY
 *   3) FLASH_LATENCY_4 (480MHz@VOS0 = 4 wait states)
 *   4) AHB /2=240MHz, APBx /2=120MHz (总线频率上限)
 *
 *   HSE_VALUE 必须与板载晶振一致: NUCLEO-H743ZI = 25MHz
 *   (见 Inc/stm32h7xx_hal_conf.h 的 HSE_VALUE 定义)
 * ================================================================ */
static void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    /* --- 1. PWR: LDO 供电 + VOS0 (必须在 RCC PLL 之前) --- */
    HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
    HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE0);
    while (__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY) == RESET) {
    }

    /* --- 2. HSE + PLL1: 25MHz -> 480MHz --- */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState       = RCC_HSE_ON;
    RCC_OscInitStruct.PLL.PLLState   = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM       = 5;      /* 25MHz/5 = 5MHz ref */
    RCC_OscInitStruct.PLL.PLLN       = 192;    /* VCO = 960MHz */
    RCC_OscInitStruct.PLL.PLLP       = 2;      /* SYSCLK = 480MHz */
    RCC_OscInitStruct.PLL.PLLQ       = 4;      /* 240MHz (备用) */
    RCC_OscInitStruct.PLL.PLLR       = 2;      /* 480MHz (备用) */
    RCC_OscInitStruct.PLL.PLLRGE     = RCC_PLL1VCIRANGE_2;  /* ref 4-8MHz */
    RCC_OscInitStruct.PLL.PLLVCOSEL  = RCC_PLL1VCOWIDE;     /* VCO 192-960MHz */
    RCC_OscInitStruct.PLL.PLLFRACN   = 0;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }

    /* --- 3. 总线分频: SYSCLK 480 / AHB 240 / APB 120 --- */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2
                                | RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
    RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.SYSCLKDivider  = RCC_SYSCLK_DIV1;   /* 480 */
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_HCLK_DIV2;     /* 240 */
    RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;     /* 120 */
    RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;     /* 120 (USART3) */
    RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;     /* 120 */
    RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;     /* 120 */
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) {
        Error_Handler();
    }

    /* 时钟切换后按新频率刷新 SysTick (否则 HAL 超时按 64MHz 计, 实际快 7.5 倍) */
    HAL_InitTick(TICK_INT_PRIORITY);

    SystemCoreClockUpdate();
}

/* ================================================================
 * GPIO 初始化 (PA0: 推理耗时测量引脚)
 * ================================================================ */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();

    GPIO_InitStruct.Pin   = PROFILE_PIN;
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull  = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    HAL_GPIO_Init(PROFILE_PORT, &GPIO_InitStruct);
}

/* ================================================================
 * CPU L1-Cache 使能
 * ================================================================ */
static void CPU_CACHE_Enable(void)
{
    SCB_EnableICache();
    SCB_EnableDCache();
}

/* ================================================================
 * 错误处理: 红灯常亮
 * ================================================================ */
static void Error_Handler(void)
{
    BSP_LED_On(LED3);
    while (1) {
    }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
    (void)file;
    (void)line;
    while (1) {
    }
}
#endif
