#!/usr/bin/env python3
"""综合验证: 复位板子 -> 抓启动信息 -> 发送音频 PCM -> 等待 RESULT

用法:
    python reset_and_test.py COM19 [音频路径] [--timeout 60]
"""
import argparse
import subprocess
import sys
import time

import serial

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
import send_pcm


def main():
    ap = argparse.ArgumentParser(description="复位板子并做一帧完整链路测试")
    ap.add_argument("port", help="串口号, 如 COM19")
    ap.add_argument("audio", nargs="?", default=None, help="音频文件(默认 test_voice_32k.wav)")
    ap.add_argument("--timeout", type=float, default=90.0, help="等待 RESULT 超时秒数")
    args = ap.parse_args()

    audio = args.audio or (__file__.rsplit("\\", 1)[0] + "\\test_voice_32k.wav")
    ser = serial.Serial(args.port, 115200, timeout=0.2)
    print(f"[PC] 已打开 {args.port} @ 115200")

    # 1. 复位板子 (pyocd reset 后目标自动运行)
    print("[PC] 复位板子...")
    subprocess.run(
        ["python", "-m", "pyocd", "commander", "-t", "stm32h743xx", "-c", "reset"],
        capture_output=True,
        check=False,
    )

    # 2. 抓启动信息, 直到出现 "waiting for PCM" 或超时
    print("[PC] 等待板端启动完成 (waiting for PCM)...")
    buf, ready, t0 = b"", False, time.time()
    while time.time() - t0 < 20:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode("utf-8", errors="replace").rstrip("\r")
                print(s)
                if "waiting for PCM" in s:
                    ready = True
        else:
            time.sleep(0.05)
        if ready:
            break
    if not ready:
        print("[PC] 警告: 未捕获到 waiting for PCM, 仍尝试发送 (板端可能已就绪)")

    # 3. 发送音频
    pcm = send_pcm.load_wav(audio) if audio.lower().endswith(".wav") else send_pcm.load_pcm(audio)
    print(f"[PC] PCM: {len(pcm)} B ({send_pcm.PCM_SAMPLES} samples @32kHz = {send_pcm.PCM_SAMPLES / 32000:.3f}s)")
    ser.write(send_pcm.MAGIC + pcm)
    ser.flush()
    print(f"[PC] -> 发送 {len(send_pcm.MAGIC) + len(pcm)} B, 等待处理...")

    # 4. 等待 RESULT
    t0, got = time.time(), False
    while time.time() - t0 < args.timeout:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode("utf-8", errors="replace").rstrip("\r")
                print(s)
                if s.startswith("RESULT"):
                    got = True
                    break
        else:
            time.sleep(0.05)
        if got:
            break

    ser.close()
    if not got:
        print("[PC] 超时未收到 RESULT")
        sys.exit(1)


if __name__ == "__main__":
    main()
