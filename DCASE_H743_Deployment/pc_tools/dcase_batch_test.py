#!/usr/bin/env python3
"""DCASE 10 类场景批量实测: 每类 1 个 wav -> 32k/16bit -> 串口发送 -> 汇总正确率.

数据来源: dcase_fetch.py 提取的 dataset/samples/<scene>.wav (44.1kHz/24bit 1s)
转换: 重采样 32kHz + 循环补齐到 35072 样本 (板端 LOGMEL_PCM_LEN)

用法: python dcase_batch_test.py COM19 [--samples dataset/samples] [--baud 115200]
"""
import argparse
import os
import re
import struct
import sys
import time

import numpy as np
import serial
from scipy.io import wavfile
from scipy.signal import resample_poly

MAGIC = bytes([0xA5, 0x5A, 0xA5, 0x5A])
PCM_SAMPLES = 35072          # LOGMEL_PCM_LEN
SR_TARGET = 32000

CLASSES = ["airport", "bus", "metro", "metro_station", "park",
           "public_square", "shopping_mall", "street_pedestrian",
           "street_traffic", "tram"]


def load_wav_32k(path):
    """读任意 wav -> 32kHz int16, 循环补齐到 PCM_SAMPLES."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    # 24-bit (wavfile 返回 int32, 值域 ±2^23) 与 32-bit int 区分归一化
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        scale = 2147483648.0 if np.abs(data).max() > (1 << 24) else 8388608.0
        x = data.astype(np.float64) / scale
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    else:
        x = data.astype(np.float64)

    # 重采样到 32kHz (poly 因子化简 44100->32000 = 320/441)
    if rate != SR_TARGET:
        x = resample_poly(x, SR_TARGET, rate)

    # 循环补齐到 PCM_SAMPLES
    n = len(x)
    if n < PCM_SAMPLES:
        rep = int(np.ceil(PCM_SAMPLES / n))
        x = np.tile(x, rep)[:PCM_SAMPLES]
    else:
        x = x[:PCM_SAMPLES]

    pcm = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
    return pcm.tobytes()


def send_once(ser, scene, pcm, timeout=30.0):
    ser.reset_input_buffer()
    time.sleep(0.3)
    ser.write(MAGIC + pcm)
    ser.flush()
    buf = b""
    t0 = time.time()
    while time.time() - t0 < timeout:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode("utf-8", errors="replace").rstrip("\r")
                if s.startswith("RESULT"):
                    return s
    return None


def main():
    ap = argparse.ArgumentParser(description="DCASE 10 类场景板端实测")
    ap.add_argument("port", help="串口 (ST-Link VCP, 如 COM19)")
    ap.add_argument("--samples", default=os.path.join(os.path.dirname(__file__), "dataset", "samples"))
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    files = [(c, os.path.join(args.samples, f"{c}.wav")) for c in CLASSES]
    missing = [c for c, p in files if not os.path.exists(p)]
    if missing:
        print(f"缺少样本: {missing}"); sys.exit(1)

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    print(f"[PC] {args.port} @ {args.baud}, 共 {len(files)} 个样本")

    csv_path = os.path.join(os.path.dirname(__file__), "batch_result.csv")
    with open(csv_path, "w", encoding="utf-8") as fout:
        fout.write("scene,expected_class,pred,conf,ok\n")
        results = []
        for scene, path in files:
            pcm = load_wav_32k(path)
            print(f"\n=== {scene} ({os.path.basename(path)}) ===", flush=True)
            line = send_once(ser, scene, pcm)
            if line is None:
                print("  [TIMEOUT] 未收到 RESULT", flush=True)
                fout.write(f"{scene},{CLASSES.index(scene)},-,-,timeout\n")
                fout.flush()
                results.append((scene, None, 0.0))
                continue
            # RESULT class=4,park,conf=0.831,dsp=...,inf=...,total=...
            m = re.search(r"class=(\d+),([a-z_]+),conf=([0-9.eE+-]+)", line)
            if m is None:
                print(f"  [PARSE] 无法解析: {line}", flush=True)
                fout.write(f"{scene},{CLASSES.index(scene)},-,-,parse_err\n")
                fout.flush()
                results.append((scene, None, 0.0))
                continue
            pred, conf = m.group(1), float(m.group(3))
            ok = (pred == str(CLASSES.index(scene)))
            print(f"  {line}", flush=True)
            print(f"  {'[OK] 正确' if ok else '[X] 错误'}  (期望 class={CLASSES.index(scene)}/{scene})", flush=True)
            fout.write(f"{scene},{CLASSES.index(scene)},{pred},{conf:.4f},{'OK' if ok else 'X'}\n")
            fout.flush()
            results.append((scene, pred, conf, ok))

        print("\n================ 汇总 ================", flush=True)
        correct = sum(1 for r in results if len(r) == 4 and r[3])
        for r in results:
            if len(r) == 3:
                scene, pred, conf = r
                print(f"  {scene:18s} -> 超时/失败", flush=True)
            else:
                scene, pred, conf, ok = r
                print(f"  {scene:18s} -> pred={pred:>3}  conf={conf:.3f}  {'OK' if ok else 'X'}", flush=True)
        print(f"正确率: {correct}/{len(results)} = {correct/len(results)*100:.0f}%", flush=True)
    ser.close()


if __name__ == "__main__":
    main()
