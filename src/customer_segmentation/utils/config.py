from pathlib import Path
from typing import Any, Dict

import yaml

from .exceptions import ConfigurationError


class Config:
    """Load and access project YAML configuration."""

    def __init__(self, config_dir: str = "configs"):
        self.project_root = Path(__file__).resolve().parents[3]
        self.config_dir = self.project_root / config_dir
        self._configs: Dict[str, Dict[str, Any]] = {}

        self._load_configs()

    def _load_configs(self) -> None:
        """Load all YAML configuration files."""
        if not self.config_dir.exists():
            raise ConfigurationError(
                f"Configuration directory not found: {self.config_dir}"
            )

        for path in self.config_dir.glob("*.yaml"):
            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}

            if len(data) == 1:
                key, value = next(iter(data.items()))
                self._configs[key] = value
            else:
                self._configs[path.stem] = data

    def get_config(self, name: str) -> Dict[str, Any]:
        """Return a named configuration section."""
        return self._configs.get(name, {})

    def get(self, key: str, default: Any = None) -> Any:
        """Search loaded configs for a key."""
        for config in self._configs.values():
            if key in config:
                return config[key]

        return default


_config = None


def get_config() -> Config:
    """Return singleton configuration instance."""
    global _config

    if _config is None:
        _config = Config()

    return _config
