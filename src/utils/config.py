"""Configuration loader for the project."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

# Load .env
load_dotenv(CONFIGS_DIR / ".env")


def get_api_key() -> str:
    """Get the dormant puzzle API key."""
    key = os.getenv("DORMANT_API_KEY")
    if not key:
        raise ValueError("DORMANT_API_KEY not set. Check configs/.env")
    return key


def load_config() -> dict:
    """Load the main YAML config."""
    config_path = CONFIGS_DIR / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_config(model_name: str) -> dict:
    """Get config for a specific model."""
    config = load_config()
    if model_name in config["models"]:
        return config["models"][model_name]
    # Try matching by name field
    for key, val in config["models"].items():
        if val["name"] == model_name:
            return val
    raise KeyError(f"Model {model_name} not found in config")
