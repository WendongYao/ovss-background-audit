"""Parallel range download of PyTorch-Encoding's processed PASCAL-Context val masks (val.pth, from the official
trainval_merged.json via the Detail API; 60 classes, 0 = background), with resume and S3 multipart-ETag check.
Usage: python p5_get.py OUT_PATH"""

import concurrent.futures as cf
import hashlib
import os
import sys
import time
import urllib.request

URL = "https://hangzh.s3.amazonaws.com/encoding/data/pcontext/val.pth"
SIZE = 913687510
ETAG = "edb3cf170802955cdb78cb6d6e8a57dc-55"
PART = 16 * 1024 * 1024  # S3 multipart part size consistent with 55 parts
OUT = sys.argv[1]
TMP = OUT + ".parts"
os.makedirs(TMP, exist_ok=True)
n_parts = (SIZE + PART - 1) // PART
assert n_parts == 55, n_parts


def get(i):
    p = f"{TMP}/{i:03d}"
    lo, hi = i * PART, min(SIZE, (i + 1) * PART) - 1
    if os.path.exists(p) and os.path.getsize(p) == hi - lo + 1:
        return i, "cached"
    for attempt in range(20):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == hi - lo + 1:
                open(p + ".tmp", "wb").write(data)
                os.replace(p + ".tmp", p)
                return i, f"ok attempt {attempt}"
        except Exception as e:  # noqa: BLE001
            time.sleep(3 + attempt)
    return i, "FAILED"


t0 = time.time()
with cf.ThreadPoolExecutor(16) as ex:
    for i, s in ex.map(get, range(n_parts)):
        print(f"[{time.time() - t0:7.0f}s] part {i} {s}", flush=True)
md5s = []
with open(OUT, "wb") as f:
    for i in range(n_parts):
        d = open(f"{TMP}/{i:03d}", "rb").read()
        md5s.append(hashlib.md5(d).digest())
        f.write(d)
etag = hashlib.md5(b"".join(md5s)).hexdigest() + f"-{n_parts}"
print("size", os.path.getsize(OUT), "etag", etag, "match", etag == ETAG and os.path.getsize(OUT) == SIZE, flush=True)
