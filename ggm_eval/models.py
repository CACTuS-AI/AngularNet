"""Released models."""
MODELS = {
    "tinyimagenet": {
        "resnet18_ggm_1bit_N1":    dict(arch="resnet18", bits="1/1",       N=1),
        "resnet18_ggm_1.58bit_N1": dict(arch="resnet18", bits="1.58/1.58", N=1),
        "resnet34_ggm_1bit_N1":    dict(arch="resnet34", bits="1/1",       N=1),
        "resnet34_ggm_1.58bit_N1": dict(arch="resnet34", bits="1.58/1.58", N=1),
    },
    "imagenet": {
        "resnet18_ggm_1bit_N1": dict(arch="resnet18", bits="1/1", N=1),
        "resnet34_ggm_1bit_N1": dict(arch="resnet34", bits="1/1", N=1),
    },
    "cifar10": {
        "resnet18_ggm_1bit_N1": dict(arch="resnet18", bits="1/1", N=1),
        "vgg16_ggm_1bit_N1":    dict(arch="vgg16",    bits="1/1", N=1),
    },
    "adftd": {
        "medformer_ggm_1bit": dict(arch="Medformer", bits="1/1", N=1),
    },
    "ptbxl": {
        "medformer_ggm_1bit": dict(arch="Medformer", bits="1/1", N=1),
    },
    "sst2": {
        "bert_ggm_1bit_N1": dict(arch="BERT-base", bits="1/1", N=1),
    },
    "qnli": {
        "bert_ggm_1bit_N1": dict(arch="BERT-base", bits="1/1", N=1),
    },
}


def resolve(dataset: str, name: str) -> str:
    """Case-insensitive lookup that also accepts '-' for '_'."""
    key = name.strip().lower().replace("-", "_").removesuffix(".pt")
    for m in MODELS[dataset]:
        if m.lower() == key:
            return m
    raise KeyError(name)
