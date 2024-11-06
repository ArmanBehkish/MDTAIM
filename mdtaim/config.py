"""Handles the configuration"""

from typing import Dict, Any, List
import yaml


class Config:
    """
    Handles the configuration,
    config.yaml is in the config/ directory
    """

    def __init__(self, logger, config_path="./config/config.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.logger = logger

    def load_config(self) -> None:
        """Load the configuration from the YAML file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.critical(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            self.logger.critical(f"Error parsing YAML file: {e}")
        self.logger.info("Config loaded successfully!")

    def __str__(self) -> str:
        # return in a separate line for each key-value pair
        return "\n".join([f"-={k}=-: {v}" for k, v in self._config.items()])

    def get_config(self) -> Dict[str, Any]:
        """Return the loaded configuration."""
        return self._config

    def update_config(self, keys: List[str], value: Any) -> None:
        """
        update the config value for the given hierarchy of keys
        example:
        config.update_config(["logging", "log_file_prefix"], "prod")
        """
        config = self._config
        # get the value for the key k
        for key in keys[:-1]:
            try:
                config = config[key]
            except KeyError as e:
                self.logger.error(f"Key {key} does not exist!")
                raise KeyError(f"Key {key} does not exist!") from e
        if keys[-1] not in config:
            self.logger.error(f"Key {keys[-1]} does not exist!")
            raise KeyError(f"Key {keys[-1]} does not exist!")
        else:
            config[keys[-1]] = value
            self.logger.info(f"{keys[-1]} updated!")
        self.save_config()

    def save_config(self) -> None:
        """Save the current configuration back to the YAML file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False)
                self.logger.info(f"Config saved to {self.config_path}")
        except IOError as e:
            self.logger.error(f"Error saving config file: {e}")
            raise IOError(f"Error saving config file: {e}") from e
