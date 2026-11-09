"""Released models."""
MODELS = {
    "resnet18_ggm_1bit_N1": dict(arch="resnet18", bits="1/1", N=1),
    "vgg16_ggm_1bit_N1": dict(arch="vgg16", bits="1/1", N=1),
}


def resolve(name: str) -> str:
    """Case-insensitive lookup that also accepts '-' for '_'."""
    key = name.strip().lower().replace("-", "_").removesuffix(".pt")
    for m in MODELS:
        if m.lower() == key:
            return m
    raise KeyError(name)
