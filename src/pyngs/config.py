"""Singleton configuration with CLI parsing, YAML support, attribute access,
and automatic __init__ injection via @Config.configurable."""

import yaml
import argparse
import inspect
import functools
from typing import Any, Dict, Optional, Type


class Config:
    """Singleton config with argparse, YAML, and attribute-style access.

    Access values via dot notation (cfg.lr), brackets (cfg["lr"]), or .get().
    """

    _instance: Optional['Config'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.config: Dict[str, Any] = {}
        self._parser = argparse.ArgumentParser()
        self._parsed = False
        self._registered_prefixes: set = set()

    # -- Singleton --------------------------------------------------------

    @classmethod
    def instance(cls) -> 'Config':
        return cls()

    @classmethod
    def reset(cls) -> 'Config':
        cls._instance = None
        return cls()

    # -- Attribute / item access ------------------------------------------

    def __getattr__(self, key: str) -> Any:
        config = self.__dict__.get('config')
        if config is None:
            raise AttributeError(key)
        try:
            return config[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.config

    # -- Registration -----------------------------------------------------

    def add_argument(self, name: str, *args, **kwargs):
        self._parser.add_argument(
            name, *args, **kwargs,
        )

    # -- Parsing ----------------------------------------------------------

    def parse_cli_args(self, args=None):
        parsed = self._parser.parse_args(args)
        self.config.update(vars(parsed))
        self._parsed = True

    # -- YAML -------------------------------------------------------------

    def load_from_yaml(self, path: str):
        try:
            with open(path) as f:
                self.config.update(yaml.safe_load(f) or {})
        except FileNotFoundError:
            print(f"Warning: Config file not found: {path}")

    def save_to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    # -- Access helpers ---------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def update(self, updates: Dict[str, Any]):
        self.config.update(updates)

    # -- @configurable decorator ------------------------------------------

    @staticmethod
    def configurable(cls=None, *, prefix: Optional[str] = None):
        """Class decorator that auto-registers __init__ params as --<prefix>.<param> CLI args.
        At instantiation, missing arguments are filled from the parsed config."""

        def wrap(klass: Type) -> Type:
            _prefix = prefix if prefix is not None else klass.__name__.lower()
            config = Config.instance()

            if _prefix in config._registered_prefixes:
                raise ValueError(f"Config prefix '{_prefix}' already registered.")
            config._registered_prefixes.add(_prefix)

            sig = inspect.signature(klass.__init__)
            param_names = []

            for name, param in sig.parameters.items():
                if name == 'self' or param.kind in (
                    inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                param_names.append(name)
                full_name = f"{_prefix}.{name}"
                default = param.default if param.default is not inspect.Parameter.empty else None
                config._parser.add_argument(f"--{full_name}", default=default)
                config.config[full_name] = default

            original_init = klass.__init__

            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                bound = sig.bind_partial(self, *args, **kwargs)
                for name in param_names:
                    if name not in bound.arguments:
                        value = config.get(f"{_prefix}.{name}")
                        if value is not None:
                            kwargs[name] = value
                original_init(self, *args, **kwargs)

            klass.__init__ = new_init
            klass._config_prefix = _prefix
            return klass

        if cls is not None:
            return wrap(cls)
        return wrap
