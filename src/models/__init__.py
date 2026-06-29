"""M1 模型特征提取包。

适配器在子模块中按需导入（避免本地无 torch 时 import 失败）。
"""

ADAPTER_REGISTRY = {
    "pythia": ("src.models.pythia_adapter", "PythiaAdapter"),
    "rwkv": ("src.models.rwkv_adapter", "RWKVAdapter"),
    "mamba": ("src.models.mamba_adapter", "MambaAdapter"),
    "awd_lstm": ("src.models.awd_lstm_adapter", "AWDLSTMAdapter"),
}


def get_adapter(name: str, **kwargs):
    """按名称惰性实例化适配器（只有用到时才 import 重型依赖）。"""
    import importlib

    if name not in ADAPTER_REGISTRY:
        raise KeyError(f"未知模型 {name}，可选: {list(ADAPTER_REGISTRY)}")
    module_path, cls_name = ADAPTER_REGISTRY[name]
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**kwargs)
