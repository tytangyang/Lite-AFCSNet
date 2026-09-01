#!/usr/bin/env python3
"""板端 int8 网络 vs PC onnxruntime(QDQ) 概率分布对比 (纯 dump 模式).

板端 dump 模式回传 3 个 blob (均带 4B 长度头):
  [0] FFT 快照 4096 float
  [1] logmel  256x65 float
  [2] 反量化 logits / softmax 概率 10 float
PC 端: 板端 logmel 喂 ONNX -> PC softmax 概率, 与板端 logits 对比.

用法: python compare_logits.py COM19
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import serial
from scipy.io import wavfile
from scipy.signal import resample_poly

SRC_DIR = os.path.join(os.path.dirname(__file__), "dataset", "samples")
ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "asc_base_p0_65_int8_qdq.onnx")

DUMP_MAGIC = bytes([0x5A, 0xA5, 0x5A, 0xA5])
PCM_LEN = 35072
N_MELS, N_FRAMES = 256, 65

CLASSES = ["airport", "bus", "metro", "metro_station", "park",
           "public_square", "shopping_mall", "street_pedestrian",
           "street_traffic", "tram"]


def load_wav_32k(path):
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    else:
        scale = 2147483648.0 if np.abs(data).max() > (1 << 24) else 8388608.0
        x = data.astype(np.float64) / scale
    if rate != 32000:
        x = resample_poly(x, 32000, rate)
    if x.size < PCM_LEN:
        x = np.tile(x, int(np.ceil(PCM_LEN / x.size)))
    x = x[:PCM_LEN]
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2").tobytes()


def recv_blob(ser, timeout_s=40):
    hdr = b""
    t0 = time.time()
    while len(hdr) < 4:
        hdr += ser.read(4 - len(hdr))
        if time.time() - t0 > timeout_s:
            raise TimeoutError("长度头超时")
    n = int.from_bytes(hdr, "little")
    data = b""
    t0 = time.time()
    while len(data) < n:
        data += ser.read(n - len(data))
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"数据超时 (收到 {len(data)}/{n})")
    return data


def dump_all(ser, pcm):
    """发 DUMP_MAGIC -> 收 3 个 blob, 返回 (logmel[256,65], board_logits[10])."""
    ser.reset_input_buffer()
    ser.write(DUMP_MAGIC + pcm)
    ser.flush()
    recv_blob(ser)   # FFT 快照 (不需要)
    lm = np.frombuffer(recv_blob(ser), dtype="<f4").copy().reshape(N_MELS, N_FRAMES)
    logits = np.frombuffer(recv_blob(ser), dtype="<f4").copy()   # [10] 概率
    return lm, logits


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM19"
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    ser = serial.Serial(port, 115200, timeout=2)
    time.sleep(0.5)

    print(f"{'scene':18s} {'board':>14s} {'PC':>14s}  {'一致':3s}  "
          f"corr  max|Δp|")
    rows = []
    for c in CLASSES:
        pcm = load_wav_32k(os.path.join(SRC_DIR, f"{c}.wav"))
        try:
            lm, b_probs = dump_all(ser, pcm)
        except TimeoutError as e:
            print(f"{c:18s}  [TIMEOUT] {e}")
            continue
        pc_logits = sess.run(None, {in_name: lm[None, None]})[0][0]
        pc_probs = softmax(pc_logits)
        b_probs = softmax(b_probs)   # 板端返回的已是概率, 再归一化无害

        b_cls, pc_cls = int(b_probs.argmax()), int(pc_probs.argmax())
        agree = "OK" if b_cls == pc_cls else "X"
        corr = np.corrcoef(b_probs, pc_probs)[0, 1]
        maxdp = np.abs(b_probs - pc_probs).max()
        print(f"{c:18s} {CLASSES[b_cls]:>14s} {CLASSES[pc_cls]:>14s}  "
              f"{agree:3s}  {corr:.3f}  {maxdp:.4f}  "
              f"board[{CLASSES[b_cls]}]{b_probs[b_cls]:.3f} vs pc[{CLASSES[pc_cls]}]{pc_probs[pc_cls]:.3f}",
              flush=True)
        rows.append((c, b_cls, pc_cls, agree, corr, maxdp))

    ser.close()
    print("\n汇总:")
    if rows:
        n_agree = sum(1 for r in rows if r[3] == "OK")
        print(f"  argmax 一致: {n_agree}/{len(rows)}")
        print(f"  概率相关度: min={min(r[4] for r in rows):.3f} "
              f"mean={np.mean([r[4] for r in rows]):.3f}")
        print(f"  max|Δp|: min={min(r[5] for r in rows):.4f} "
              f"mean={np.mean([r[5] for r in rows]):.4f} "
              f"max={max(r[5] for r in rows):.4f}")


if __name__ == "__main__":
    main()
