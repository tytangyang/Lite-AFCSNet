#!/usr/bin/env python3
"""DCASE 音频场景分类部署 - PC 端串口发送/接收脚本 (Nucleo-H743ZI2)

链路: PC 发送魔数 + 32kHz 单声道 16-bit PCM → 板端 DSP/推理 → 回传 RESULT 行

用法:
    python send_pcm.py COM5 test.wav
    python send_pcm.py COM5 audio.pcm          # 原始 16-bit PCM 亦可
    python send_pcm.py COM5 --loop test.wav    # 循环发送(连续测试)

协议(板端自同步):
    0xA5 0x5A 0xA5 0x5A + 70144 B PCM (35072 samples × int16)
    板端处理完成后回传: RESULT class=<id>,<name>,conf=<f>,dsp=<ms>,inf=<ms>,total=<ms>
"""
import argparse
import struct
import sys
import time

import serial

MAGIC = bytes([0xA5, 0x5A, 0xA5, 0x5A])
PCM_SAMPLES = 35072          # LOGMEL_PCM_LEN
PCM_BYTES = PCM_SAMPLES * 2  # 70144


def load_wav(path):
    """读取标准 PCM WAV, 返回 32kHz 单声道 16-bit 的 70144B 数据(不足补零, 超出截断)."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError(f"{path}: 不是标准 RIFF/WAVE 文件")

    pos, fmt, audio = 12, None, b''
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b'fmt ':
            fmt = body
        elif cid == b'data':
            audio = body
            break
        pos += 8 + size + (size & 1)

    if fmt is None or not audio:
        raise ValueError(f"{path}: WAV 缺少 fmt/data chunk")

    afmt, ch, rate, _, _, bps = struct.unpack('<HHIIHH', fmt[:16])
    if afmt != 1:
        raise ValueError(f"{path}: 仅支持 PCM 格式 (format={afmt})")
    if ch != 1 or rate != 32000 or bps != 16:
        raise ValueError(f"{path}: 需 32kHz 单声道 16-bit, 实际 {rate}Hz {ch}ch {bps}bit")

    n = min(len(audio) // 2, PCM_SAMPLES)
    pcm = audio[:n * 2]
    if n < PCM_SAMPLES:
        pcm += b'\x00' * (PCM_SAMPLES - n) * 2
    return pcm


def load_pcm(path):
    """读取原始 16-bit PCM(小端), 返回 70144B 数据(不足补零)."""
    with open(path, 'rb') as f:
        data = f.read()
    n = min(len(data) // 2, PCM_SAMPLES)
    pcm = data[:n * 2]
    if n < PCM_SAMPLES:
        pcm += b'\x00' * (PCM_SAMPLES - n) * 2
    return pcm


def send_once(ser, audio_path, timeout):
    """发送一帧音频并等待 RESULT. 返回 True 表示收到结果."""
    pcm = load_wav(audio_path) if audio_path.lower().endswith('.wav') else load_pcm(audio_path)
    print(f"[PC] PCM: {len(pcm)} B ({PCM_SAMPLES} samples @32kHz = {PCM_SAMPLES / 32000:.3f}s)")

    ser.reset_input_buffer()
    time.sleep(0.2)  # 等板端打印 waiting 提示

    frame = MAGIC + pcm
    ser.write(frame)
    ser.flush()
    print(f"[PC] -> 发送 {len(frame)} B (magic 4 + pcm {len(pcm)}), 等待板端处理...")

    buf = b''
    t0 = time.time()
    got_result = False
    while time.time() - t0 < timeout:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                s = line.decode('utf-8', errors='replace').rstrip('\r')
                print(s)
                if s.startswith('RESULT'):
                    got_result = True
        else:
            if got_result:
                break
            time.sleep(0.05)
    return got_result


def main():
    ap = argparse.ArgumentParser(
        description='发送音频到 Nucleo-H743ZI2 板端分类并显示结果')
    ap.add_argument('port', help='串口号, 如 COM5 (ST-Link VCP)')
    ap.add_argument('audio', help='音频文件: 32kHz 单声道 16-bit WAV, 或原始 .pcm')
    ap.add_argument('--baud', type=int, default=115200, help='波特率 (默认 115200)')
    ap.add_argument('--timeout', type=float, default=30.0, help='单帧等待结果超时秒数')
    ap.add_argument('--loop', action='store_true', help='循环发送, Ctrl+C 退出')
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    print(f"[PC] 已打开 {args.port} @ {args.baud}")

    try:
        if args.loop:
            n = 0
            while True:
                n += 1
                print(f"\n===== 第 {n} 帧 =====")
                send_once(ser, args.audio, args.timeout)
                time.sleep(0.5)
        else:
            if not send_once(ser, args.audio, args.timeout):
                print("[PC] 超时未收到 RESULT")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n[PC] 已停止")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
