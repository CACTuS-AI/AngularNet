"""Released models."""
MODELS = {
    "resnet18_ggm_1bit_N1":    dict(arch="resnet18", bits="1/1",       N=1),
    "resnet18_ggm_1.58bit_N1": dict(arch="resnet18", bits="1.58/1.58", N=1),
    "resnet18_ggm_1bit_N2":    dict(arch="resnet18", bits="1/1",       N=2),
    "resnet18_ggm_1bit_N3":    dict(arch="resnet18", bits="1/1",       N=3),
    "resnet34_ggm_1bit_N1":    dict(arch="resnet34", bits="1/1",       N=1),
    "resnet34_ggm_1.58bit_N1": dict(arch="resnet34", bits="1.58/1.58", N=1),
    "resnet34_ggm_1bit_N2":    dict(arch="resnet34", bits="1/1",       N=2),
}


def resolve(name: str) -> str:
    """Case-insensitive lookup that also accepts '-' for '_'."""
    key = name.strip().lower().replace("-", "_").removesuffix(".pt")
    for m in MODELS:
        if m.lower() == key:
            return m
    raise KeyError(name)
