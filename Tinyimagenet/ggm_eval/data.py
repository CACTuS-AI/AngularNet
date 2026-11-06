"""TinyImageNet-200 test set (the labelled val/ split): download and loading."""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import urllib.request
import zipfile

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
PREFIX = "tiny-imagenet-200/"
NUM_IMAGES = 10_000
# SHA-256 of val/val_annotations.txt in the official release; checked after a download
ANNOTATIONS_SHA256 = "c37765a8b40fce4f28962bf1847a2537e2781753974cbb2447eeb50ff31130a2"
USER_AGENT = "tinyimagenet-eval"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


# --------------------------------------------------------------------------- layout

def find_dataset_dir(root: str) -> str | None:
    """`root` may be the tiny-imagenet-200 directory itself or its parent."""
    for cand in (root, os.path.join(root, "tiny-imagenet-200")):
        if os.path.isfile(os.path.join(cand, "wnids.txt")) and os.path.isdir(os.path.join(cand, "val")):
            return cand
    return None


def _list_samples(d: str):
    with open(os.path.join(d, "wnids.txt")) as f:
        classes = sorted(line.strip() for line in f if line.strip())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    val = os.path.join(d, "val")
    ann = os.path.join(val, "val_annotations.txt")
    samples = []
    if os.path.isdir(os.path.join(val, "images")) and os.path.isfile(ann):   # as in the zip
        with open(ann) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    samples.append((os.path.join(val, "images", parts[0]), class_to_idx[parts[1]]))
    else:                                                                     # val/<wnid>/<file>
        for wnid in classes:
            sub = os.path.join(val, wnid)
            if os.path.isdir(sub):
                samples += [(os.path.join(sub, f), class_to_idx[wnid]) for f in sorted(os.listdir(sub))
                            if not f.endswith(".txt")]
    # fixed order: class, then file name
    return classes, sorted(samples, key=lambda x: (x[1], os.path.basename(x[0])))


def _complete(d: str | None) -> bool:
    if d is None:
        return False
    try:
        _, samples = _list_samples(d)
    except (OSError, KeyError):
        return False
    return len(samples) == NUM_IMAGES and all(os.path.isfile(p) for p, _ in samples)


# --------------------------------------------------------------------------- download

class _RangeNotSupported(Exception):
    pass


class _HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, fetched with HTTP range requests."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self.segments = url, block, 0, []
        head = self._get(0, 0)
        if head is None:
            raise _RangeNotSupported(url)
        self.size = self._total

    def _get(self, start: int, end: int, progress: str | None = None):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                return None
            self._total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            want, chunks, got = end - start + 1, [], 0
            while True:
                c = r.read(1 << 18)
                if not c:
                    break
                chunks.append(c)
                got += len(c)
                if progress:
                    print(f"\r  {progress}: {got / 1e6:6.1f} / {want / 1e6:.1f} MB", end="", flush=True)
            if progress:
                print()
            data = b"".join(chunks)
            if len(data) != want:
                raise OSError(f"short read from {self.url}: {len(data)} of {want} bytes")
            return data

    def prefetch(self, start: int, end: int, label: str):
        end = min(end, self.size - 1)
        self.segments.append((start, self._get(start, end, progress=label)))

    def _cached(self, pos: int):
        for s, data in self.segments:
            if s <= pos < s + len(data):
                return s, data
        return None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.size:
            seg = self._cached(self.pos)
            if seg is None:
                start = self.pos
                data = self._get(start, min(self.size, start + max(n - done, self.block)) - 1)
                if data is None:
                    raise _RangeNotSupported(self.url)
                self.segments.append((start, data))
                continue
            s, data = seg
            k = min(n - done, s + len(data) - self.pos)
            b[done:done + k] = data[self.pos - s:self.pos - s + k]
            done += k
            self.pos += k
        return done


def _wanted(info: zipfile.ZipInfo) -> bool:
    return info.filename.startswith(PREFIX + "val/") or info.filename == PREFIX + "wnids.txt"


def _extract(zf: zipfile.ZipFile, members, root: str):
    for i, m in enumerate(members, 1):
        zf.extract(m, root)
        if i % 1000 == 0 or i == len(members):
            print(f"\r  extracting: {i}/{len(members)} files", end="", flush=True)
    print()


def _download_with_ranges(root: str):
    f = _HttpRangeFile(URL)
    zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
    members = [m for m in zf.infolist() if _wanted(m)]
    val = [m for m in members if m.filename.startswith(PREFIX + "val/")]
    start = min(m.header_offset for m in val)
    end = max(m.header_offset + 30 + len(m.filename.encode()) + len(m.extra) + m.compress_size for m in val) + 65536
    f.prefetch(start, end, "test set (val/)")
    _extract(zf, members, root)


def _download_full_zip(root: str):
    tmp = os.path.join(root, "tiny-imagenet-200.zip.part")
    req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
        total, got = int(r.headers.get("Content-Length", 0)), 0
        while True:
            c = r.read(1 << 20)
            if not c:
                break
            out.write(c)
            got += len(c)
            print(f"\r  full zip: {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    with zipfile.ZipFile(tmp) as zf:
        _extract(zf, [m for m in zf.infolist() if _wanted(m)], root)
    os.remove(tmp)


def ensure_test_set(root: str, download: bool = True) -> str:
    """Return the tiny-imagenet-200 directory under `root`, downloading the test split if needed."""
    d = find_dataset_dir(root)
    if _complete(d):
        return d
    if not download:
        raise FileNotFoundError(f"TinyImageNet test set not found under {root!r}")
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(os.path.join(target, "val")):
        shutil.rmtree(os.path.join(target, "val"))          # incomplete earlier download
    print(f"[data] TinyImageNet test set not found; downloading it to {target}")
    try:
        _download_with_ranges(root)
    except (_RangeNotSupported, OSError, zipfile.BadZipFile, ValueError) as e:
        print(f"[data] partial download failed ({e}); downloading the full zip (~248 MB) instead")
        if os.path.isdir(os.path.join(target, "val")):
            shutil.rmtree(os.path.join(target, "val"))
        _download_full_zip(root)
    d = find_dataset_dir(root)
    if not _complete(d):
        raise RuntimeError(f"downloaded test set under {root!r} is incomplete")
    if ANNOTATIONS_SHA256:
        with open(os.path.join(d, "val", "val_annotations.txt"), "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != ANNOTATIONS_SHA256:
                raise RuntimeError("val_annotations.txt does not match the official TinyImageNet release")
    print(f"[data] test set ready: {NUM_IMAGES:,} images")
    return d


# --------------------------------------------------------------------------- dataset

class TinyImageNetTest(Dataset):
    def __init__(self, root: str, transform=test_transform):
        d = find_dataset_dir(root)
        if d is None:
            raise FileNotFoundError(f"no tiny-imagenet-200 (wnids.txt + val/) under {root!r}")
        self.dir, self.transform = d, transform
        self.classes, self.samples = _list_samples(d)
        if len(self.samples) != NUM_IMAGES:
            raise RuntimeError(f"expected {NUM_IMAGES:,} test images under {d}/val, found {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            img = im.convert("RGB")  # a few TinyImageNet images are greyscale
        return self.transform(img), label
