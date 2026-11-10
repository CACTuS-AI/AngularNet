"""Test sets."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import os
import re
import shutil
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_test_set(dataset: str, root: str, download: bool = True):
    if dataset == "tinyimagenet":
        return TinyImageNetTest(ensure_tinyimagenet(root, download=download))
    if dataset == "imagenet":
        return imagenet_test_set(root)
    if dataset == "cifar10":
        return cifar10_test_set(root, download=download)
    if dataset in GLUE:
        return GlueDevSet(dataset, ensure_glue(dataset, root, download=download))
    root = ensure_timeseries(dataset, root, download=download)
    print(f"loading the {TIMESERIES[dataset]['name']} test set from {root} ...", flush=True)
    return TimeSeriesTest(dataset, root)


# ---------------------------------------------------------------------- TinyImageNet-200

TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_PREFIX = "tiny-imagenet-200/"
TINY_NUM_IMAGES = 10_000
# SHA-256 of val/val_annotations.txt in the official release; checked after a download
TINY_ANNOTATIONS_SHA256 = "c37765a8b40fce4f28962bf1847a2537e2781753974cbb2447eeb50ff31130a2"
USER_AGENT = "tinyimagenet-eval"
tiny_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _find_tiny_dir(root: str) -> str | None:
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
    return len(samples) == TINY_NUM_IMAGES and all(os.path.isfile(p) for p, _ in samples)


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
    return info.filename.startswith(TINY_PREFIX + "val/") or info.filename == TINY_PREFIX + "wnids.txt"


def _extract(zf: zipfile.ZipFile, members, root: str):
    for i, m in enumerate(members, 1):
        zf.extract(m, root)
        if i % 1000 == 0 or i == len(members):
            print(f"\r  extracting: {i}/{len(members)} files", end="", flush=True)
    print()


def _download_with_ranges(root: str):
    f = _HttpRangeFile(TINY_URL)
    zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
    members = [m for m in zf.infolist() if _wanted(m)]
    val = [m for m in members if m.filename.startswith(TINY_PREFIX + "val/")]
    start = min(m.header_offset for m in val)
    end = max(m.header_offset + 30 + len(m.filename.encode()) + len(m.extra) + m.compress_size for m in val) + 65536
    f.prefetch(start, end, "test set (val/)")
    _extract(zf, members, root)


def _download_full_zip(root: str):
    tmp = os.path.join(root, "tiny-imagenet-200.zip.part")
    req = urllib.request.Request(TINY_URL, headers={"User-Agent": USER_AGENT})
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


def ensure_tinyimagenet(root: str, download: bool = True) -> str:
    """Return the tiny-imagenet-200 directory under `root`, downloading the test split if needed."""
    d = _find_tiny_dir(root)
    if _complete(d):
        return d
    if not download:
        raise FileNotFoundError(f"TinyImageNet test set not found under {root}")
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
    d = _find_tiny_dir(root)
    if not _complete(d):
        raise RuntimeError(f"downloaded test set under {root} is incomplete")
    if TINY_ANNOTATIONS_SHA256:
        with open(os.path.join(d, "val", "val_annotations.txt"), "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != TINY_ANNOTATIONS_SHA256:
                raise RuntimeError("val_annotations.txt does not match the official TinyImageNet release")
    print(f"[data] test set ready: {TINY_NUM_IMAGES:,} images")
    return d


class TinyImageNetTest(Dataset):
    def __init__(self, root: str, transform=tiny_transform):
        d = _find_tiny_dir(root)
        if d is None:
            raise FileNotFoundError(f"no tiny-imagenet-200 (wnids.txt + val/) under {root}")
        self.root = self.dir = d
        self.transform = transform
        self.classes, self.samples = _list_samples(d)
        if len(self.samples) != TINY_NUM_IMAGES:
            raise RuntimeError(f"expected {TINY_NUM_IMAGES:,} test images under {d}/val, found {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            img = im.convert("RGB")  # a few TinyImageNet images are greyscale
        return self.transform(img), label


# ---------------------------------------------------------------------- ImageNet-1k

IMAGENET_NUM_CLASSES, IMAGENET_NUM_IMAGES = 1000, 50_000

imagenet_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

IMAGENET_LAYOUT_HELP = (
    "Expected <data_root>/val/<wnid>/<image>.JPEG (1000 class folders, 50,000 images), or the two\n"
    "official archives ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz in <data_root>\n"
    "(they are unpacked into that layout on first use). If an earlier unpacking was interrupted, delete\n"
    "<data_root>/val and run again. Download instructions: see README (ImageNet-1k)."
)


def _has_class_folders(val_dir: str) -> bool:
    return os.path.isdir(val_dir) and sum(
        1 for e in os.scandir(val_dir) if e.is_dir()) == IMAGENET_NUM_CLASSES


def imagenet_test_set(root: str):
    val_dir = os.path.join(root, "val")
    if not _has_class_folders(val_dir):
        if os.path.isfile(os.path.join(root, "ILSVRC2012_img_val.tar")) and \
           os.path.isfile(os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")):
            print("[data] unpacking the ImageNet validation archive (one-time)")
            datasets.ImageNet(root, split="val")
        if not _has_class_folders(val_dir):
            raise FileNotFoundError(f"ImageNet validation set not found under {root}.\n{IMAGENET_LAYOUT_HELP}")
    ds = datasets.ImageFolder(val_dir, transform=imagenet_transform)  # classes and files in sorted order
    if len(ds) != IMAGENET_NUM_IMAGES:
        print(f"[data] warning: found {len(ds):,} images in {val_dir} (the full validation set has {IMAGENET_NUM_IMAGES:,})")
    return ds


# ---------------------------------------------------------------------- CIFAR-10

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR_NUM_IMAGES = 10_000

cifar_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


def cifar10_test_set(root: str, download: bool = True):
    if not download and not os.path.isdir(os.path.join(root, "cifar-10-batches-py")):
        raise FileNotFoundError(f"CIFAR-10 not found: expected {os.path.join(root, 'cifar-10-batches-py')}")
    ds = datasets.CIFAR10(root, train=False, download=download, transform=cifar_transform)
    if len(ds) != CIFAR_NUM_IMAGES:
        print(f"[data] warning: found {len(ds):,} test images (expected {CIFAR_NUM_IMAGES:,})")
    return ds


# ---------------------------------------------------------------------- ADFTD, PTB-XL

TIMESERIES = {
    "adftd": dict(name="ADFTD", channels=19, length=256, classes=("HC", "FTD", "AD"),
                  drive_id="1QcX_M58IQUBn3lDBlVVL0SDN7_QI1vWe", size="2.6 GB"),
    "ptbxl": dict(name="PTB-XL", channels=12, length=250, classes=("NORM", "MI", "STTC", "CD", "HYP"),
                  drive_id="1whskRvTZUNb1Qph2SeXEdpcU2rQY0T1E", size="3.3 GB"),
}
TEST_START = 0.8  # per class, in label.npy order, the last 20 % of subjects are the test set
DRIVE_URL = "https://drive.google.com/uc?export=download&id={}"


def _open_drive_file(file_id: str):
    """Open a public Google Drive file; large files first need the 'Download anyway' confirmation."""
    headers = {"User-Agent": USER_AGENT}
    r = urllib.request.urlopen(urllib.request.Request(DRIVE_URL.format(file_id), headers=headers), timeout=60)
    if r.headers.get_content_type() != "text/html":
        return r
    page = r.read().decode("utf-8", "replace")
    r.close()
    form = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', page)
    fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page))
    if form is None or "id" not in fields:
        raise OSError("Google Drive did not offer the file (its daily download quota may be used up; "
                      "try again later)")
    url = html.unescape(form.group(1)) + "?" + urllib.parse.urlencode(fields)
    r = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    if r.headers.get_content_type() == "text/html":
        r.close()
        raise OSError("Google Drive returned a web page instead of the file (its daily download quota "
                      "may be used up; try again later)")
    return r


def ensure_timeseries(dataset: str, root: str, download: bool = True) -> str:
    """Return `root`, downloading and unpacking the processed dataset into it if needed."""
    try:
        _timeseries_layout(root)
        return root
    except FileNotFoundError:
        if not download:
            raise
    spec = TIMESERIES[dataset]
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, f"{dataset}.zip.part")
    print(f"[data] {spec['name']} not found; downloading it ({spec['size']}) from Google Drive to {root}")
    with _open_drive_file(spec["drive_id"]) as r, open(tmp, "wb") as out:
        total, got = int(r.headers.get("Content-Length", 0)), 0
        while True:
            c = r.read(1 << 20)
            if not c:
                break
            out.write(c)
            got += len(c)
            print(f"\r  download: {got / 1e9:5.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    with zipfile.ZipFile(tmp) as zf:   # <top folder>/Feature/..., <top folder>/Label/label.npy
        members = [m for m in zf.infolist() if not m.is_dir() and len(m.filename.split("/")) == 3
                   and m.filename.split("/")[1] in ("Feature", "Label")]
        for i, m in enumerate(members, 1):
            target = os.path.join(root, *m.filename.split("/")[1:])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 500 == 0 or i == len(members):
                print(f"\r  extracting: {i}/{len(members)} files", end="", flush=True)
    print()
    os.remove(tmp)
    _timeseries_layout(root)
    return root


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _test_ids(labels: np.ndarray, num_classes: int) -> list:
    ids = []
    for c in range(num_classes):
        cls = list(labels[labels[:, 0] == c][:, 1])
        ids += cls[int(TEST_START * len(cls)):]
    return ids


def _standardize(X: np.ndarray) -> np.ndarray:
    return np.array([StandardScaler().fit_transform(x) for x in X])


def _timeseries_layout(root: str):
    feat, lab = os.path.join(root, "Feature"), os.path.join(root, "Label", "label.npy")
    if not os.path.isdir(feat) or not os.path.isfile(lab):
        raise FileNotFoundError(f"test set not found: expected {os.path.join(feat, 'feature_*.npy')} and {lab}")
    return feat, lab


class TimeSeriesTest(Dataset):
    def __init__(self, dataset: str, root: str):
        spec = TIMESERIES[dataset]
        self.root, self.classes = root, spec["classes"]
        feat_dir, label_path = _timeseries_layout(root)
        labels = np.load(label_path)
        files = sorted(os.listdir(feat_dir), key=_natural_key)
        test_ids = _test_ids(labels, len(self.classes))

        feats, ys = [], []
        if dataset == "adftd":
            files = [f for f in files if f.endswith(".npy")]
            test_ids = set(test_ids)
            for j, f in enumerate(files):
                if int(labels[j][1]) in test_ids:
                    x = np.load(os.path.join(feat_dir, f))
                    feats.append(x)
                    ys += [labels[j][0]] * len(x)
            X = _standardize(np.concatenate(feats).astype(np.float32))
        else:
            for j, f in enumerate(files):
                if j + 1 in test_ids:
                    x = np.load(os.path.join(feat_dir, f))
                    feats.append(x)
                    ys += [labels[j][0]] * len(x)
            X = _standardize(np.concatenate(feats))

        if X.shape[1:] != (spec["length"], spec["channels"]):
            raise ValueError(f"expected samples of shape ({spec['length']}, {spec['channels']}), "
                             f"got {X.shape[1:]} in {feat_dir}")
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.as_tensor(np.asarray(ys).astype(np.int64))
        self.subjects = len(feats)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ---------------------------------------------------------------------- SST-2, QNLI (GLUE)

GLUE_URL = "https://dl.fbaipublicfiles.com/glue/data/{}"
GLUE = {
    "sst2": dict(name="SST-2", archive="SST-2.zip", member="SST-2/dev.tsv", size="7 MB", examples=872,
                 classes=("0", "1"),
                 sha256="54a9c12c77e402fc14d6a66a506f24d999f596b0176eb777c972cccc2f9cc4b0"),
    "qnli": dict(name="QNLI", archive="QNLIv2.zip", member="QNLI/dev.tsv", size="11 MB", examples=5463,
                 classes=("entailment", "not_entailment"),
                 sha256="ac19409dcb04631eaaa094bdebdcf621086bc352503dfb6a1a648b614986f4fd"),
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_glue(dataset: str, root: str, download: bool = True) -> str:
    """Return the path of dev.tsv in `root`, downloading it from the GLUE archive if needed."""
    spec = GLUE[dataset]
    path = os.path.join(root, "dev.tsv")
    if os.path.isfile(path):
        return path
    if not download:
        raise FileNotFoundError(f"{spec['name']} dev set not found: expected {path}")
    os.makedirs(root, exist_ok=True)
    url, tmp = GLUE_URL.format(spec["archive"]), os.path.join(root, spec["archive"] + ".part")
    print(f"[data] {spec['name']} not found; downloading {url} ({spec['size']}) to {root}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out)
    with zipfile.ZipFile(tmp) as zf, zf.open(spec["member"]) as src, open(path + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(tmp)
    if _sha256(path + ".part") != spec["sha256"]:
        os.remove(path + ".part")
        raise RuntimeError(f"the downloaded {spec['name']} dev.tsv does not match the official GLUE file")
    os.replace(path + ".part", path)
    return path


class GlueDevSet:
    """GLUE dev split as (text_a, text_b, label) examples; encode() turns it into model inputs."""

    def __init__(self, dataset: str, path: str):
        spec = GLUE[dataset]
        self.root, self.classes = os.path.dirname(path), spec["classes"]
        label_ids = {c: i for i, c in enumerate(self.classes)}
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))[1:]
        if dataset == "sst2":   # sentence, label
            self.examples = [(r[0], None, label_ids[r[1]]) for r in rows]
        else:                   # index, question, sentence, label
            self.examples = [(r[1], r[2], label_ids[r[-1]]) for r in rows]
        if len(self.examples) != spec["examples"]:
            print(f"[data] warning: found {len(self.examples):,} examples (expected {spec['examples']:,})")

    def __len__(self):
        return len(self.examples)

    def encode(self, tokenizer, max_len: int):
        """[CLS] a [SEP] (b [SEP]), truncated to max_len (longer sequence first) and zero padded."""
        ids, segments, masks = [], [], []
        for text_a, text_b, _ in self.examples:
            a = tokenizer.tokenize(text_a)
            b = tokenizer.tokenize(text_b) if text_b else None
            if b is not None:
                while len(a) + len(b) > max_len - 3:
                    (a if len(a) > len(b) else b).pop()
            else:
                a = a[:max_len - 2]
            tokens = ["[CLS]"] + a + ["[SEP]"]
            seg = [0] * len(tokens)
            if b:
                tokens += b + ["[SEP]"]
                seg += [1] * (len(b) + 1)
            pad = max_len - len(tokens)
            ids.append(tokenizer.convert_tokens_to_ids(tokens) + [0] * pad)
            segments.append(seg + [0] * pad)
            masks.append([1] * len(tokens) + [0] * pad)
        labels = [e[2] for e in self.examples]
        return TensorDataset(torch.tensor(ids), torch.tensor(segments), torch.tensor(masks), torch.tensor(labels))
