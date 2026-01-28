import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认的权重加载器。

    使用方法：
    通常不需要手动调用，由load_model自动使用

    参数：
    - param: 模型参数
    - loaded_weight: 从文件加载的权重张量

    功能：
    直接将加载的权重复制到模型参数中
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    加载模型权重。

    使用方法：
    load_model(model, "/path/to/model")

    参数：
    - model: 要加载权重的模型
    - path: 模型权重目录路径

    支持的格式：
    - safetensors格式（.safetensors文件）

    特殊处理：
    1. 支持packed modules（将多个权重打包到一个文件中）
    2. 支持tensor parallelism（自动分片）
    3. 支持自定义weight_loader（用于特殊处理）

    使用场景：
    1. 模型初始化后加载权重
    2. 支持Qwen等模型的packed格式
    3. 分布式权重加载

    权重映射：
    - 检查模型的packed_modules_mapping属性
    - 将权重名称映射到正确的参数位置
    - 处理分片和并行加载
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 遍历所有safetensors文件
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的所有权重
            for weight_name in f.keys():
                # 检查是否为packed module
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 权重映射
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)

                        # 获取模型参数
                        param = model.get_parameter(param_name)

                        # 获取权重加载器
                        weight_loader = getattr(param, "weight_loader")

                        # 加载权重
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通权重（非packed）
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))
