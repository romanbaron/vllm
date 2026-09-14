# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.tracing import instrument

_MODELEXPRESS_LOADER_MODULE = "modelexpress.engines.vllm.loader"
_MISSING_MODELEXPRESS_MODULES = frozenset(
    {
        "modelexpress",
        "modelexpress.engines",
        "modelexpress.engines.vllm",
        _MODELEXPRESS_LOADER_MODULE,
    }
)


def _missing_modelexpress_error() -> ImportError:
    return ImportError(
        "The 'modelexpress' load format requires the ModelExpress Python package. "
        "Install it with `pip install modelexpress`."
    )


class ModelExpressModelLoader(BaseModelLoader):
    """Thin vLLM loader wrapper for ModelExpress."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._loader = self._load_modelexpress_loader(load_config)

    @staticmethod
    def _load_modelexpress_loader(load_config: LoadConfig) -> BaseModelLoader:
        try:
            module = importlib.import_module(_MODELEXPRESS_LOADER_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name not in _MISSING_MODELEXPRESS_MODULES:
                raise
            raise _missing_modelexpress_error() from exc

        ModelExpressVllmLoader = module.MxModelLoader
        return ModelExpressVllmLoader(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        self._loader.download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        self._loader.load_weights(model, model_config)

    def on_sleep(self, level: int) -> None:
        self._loader.on_sleep(level)

    def on_wake_up(self, tags: list[str] | None) -> None:
        self._loader.on_wake_up(tags)

    def on_weights_reloaded(self) -> None:
        self._loader.on_weights_reloaded()

    def reload_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        model: nn.Module,
        prefix: str = "",
    ) -> nn.Module:
        return self._loader.reload_model(vllm_config=vllm_config, model_config=model_config, model=model, prefix=prefix)

    @instrument(span_name="Load model")
    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        model = self._loader.load_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )
        return model.eval()

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
        vllm_config: VllmConfig,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        model = self.reload_model(vllm_config, model_config, model)
        return self.module_to_named_tensors(model)

    def module_to_named_tensors(self, module: nn.Module) -> Generator[tuple[str, torch.Tensor], None, None]:
        for name, tensor in module.named_parameters():
            if tensor.is_meta:
                continue
            yield name, tensor
