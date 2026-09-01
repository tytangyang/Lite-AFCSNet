#!/usr/bin/env python3
"""从 Zenodo dev 集 zip 分卷提取 10 类场景各 1 个 wav（HTTP Range, 总下载约 2-3MB）.

每个分卷只含单一场景。策略:
  1. 对每卷: 下载末尾 64KB 定位 EOCD -> cd_off/cd_size
  2. 只下载 CD 开头 1KB, 解析首条目的文件名 -> 识别该卷场景
  3. 10 类集齐后, 对选中卷: 下载 local header + 压缩数据 -> zlib 解压

依赖: 仅标准库. 用法: python dcase_fetch.py [--limit K] [--out DIR]
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

RECORD_DEV = 6337421          # TAU Urban Acoustic Scenes 2022 Mobile, Development
API = f"https://zenodo.org/api/records/{RECORD_DEV}"

CLASSES = ["airport", "bus", "metro", "metro_station", "park",
           "public_square", "shopping_mall", "street_pedestrian",
           "street_traffic", "tram"]

EOCD_SIG = b"PK\x05\x06"
CD_SIG = b"PK\x01\x02"


def http_get(url, start=None, end=None, retries=5):
    headers = {"User-Agent": "dcase-fetch/1.0"}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return b""


def get_eocd(url, total):
    tail = http_get(url, total - 65536, total - 1)
    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise RuntimeError("EOCD not found")
    e = tail[idx:]
    return struct.unpack("<I", e[16:20])[0], struct.unpack("<I", e[12:16])[0]


def parse_cd_head(cd: bytes):
    """Parse CD bytes; return list of (name, method, csize, usize, lho)."""
    out, off = [], 0
    while off + 46 <= len(cd):
        if cd[off:off + 4] != CD_SIG:
            break
        method = struct.unpack("<H", cd[off + 10:off + 12])[0]
        csize = struct.unpack("<I", cd[off + 20:off + 24])[0]
        usize = struct.unpack("<I", cd[off + 24:off + 28])[0]
        nlen = struct.unpack("<H", cd[off + 28:off + 30])[0]
        elen = struct.unpack("<H", cd[off + 30:off + 32])[0]
        clen = struct.unpack("<H", cd[off + 32:off + 34])[0]
        lho = struct.unpack("<I", cd[off + 42:off + 46])[0]
        name = cd[off + 46:off + 46 + nlen].decode("utf-8", "replace")
        out.append({"name": name, "method": method, "csize": csize,
                    "usize": usize, "lho": lho})
        off += 46 + nlen + elen + clen
    return out


def entry_scene(entry):
    base = os.path.basename(entry["name"])
    return base.split("-")[0] if "-" in base else None


def fetch_wav(url, entry):
    lh = http_get(url, entry["lho"], entry["lho"] + 29)
    if len(lh) < 30:
        raise RuntimeError(f"bad local header {entry['name']}")
    nlen = struct.unpack("<H", lh[26:28])[0]
    elen = struct.unpack("<H", lh[28:30])[0]
    data_off = entry["lho"] + 30 + nlen + elen
    raw = http_get(url, data_off, data_off + entry["csize"] - 1)
    if len(raw) != entry["csize"]:
        raise RuntimeError(f"short read {len(raw)}/{entry['csize']} {entry['name']}")
    if entry["method"] == 0:
        return raw
    if entry["method"] == 8:
        return zlib.decompress(raw, -15)
    raise RuntimeError(f"unsupported method {entry['method']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只取前 K 类 (调试)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dataset", "samples"))
    args = ap.parse_args()

    r = json.loads(http_get(API).decode("utf-8"))
    vols = [(x["key"], x["links"]["self"], x["size"])
            for x in r["files"] if "audio." in x["key"]]
    print(f"[1/4] 共 {len(vols)} 个分卷, 扫描场景归属...")

    chosen = {}   # scene -> (url, entry)
    order = []
    for key, url, total in vols:
        cd_off, cd_size = get_eocd(url, total)
        head = http_get(url, cd_off, cd_off + 1023)
        entries = parse_cd_head(head)
        scene = None
        for e in entries:
            scene = entry_scene(e)
            if scene:
                break
        tag = scene if scene else "?"
        print(f"      {key.split('audio.')[1].split('.zip')[0]:>2}: {tag}  "
              f"({total/1e6:.0f} MB, cd={cd_size/1e6:.2f} MB)")
        if scene in CLASSES and scene not in chosen:
            chosen[scene] = (url, e)
            order.append(scene)
        if args.limit and len(chosen) >= args.limit:
            break

    missing = [c for c in CLASSES if c not in chosen]
    print(f"[2/4] 集齐 {len(chosen)}/{len(CLASSES)} 类; 缺失: {missing}")

    os.makedirs(args.out, exist_ok=True)
    print("[3/4] 下载 10 个 wav...")
    for scene in order:
        url, e = chosen[scene]
        data = fetch_wav(url, e)
        path = os.path.join(args.out, f"{scene}.wav")
        with open(path, "wb") as fp:
            fp.write(data)
        print(f"      {os.path.basename(e['name']):60s} -> {scene}.wav ({len(data)/1000:.1f} KB)")
    print(f"[4/4] 完成: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
