from tomllib import load
from pydantic import BaseModel


class Config(BaseModel):
    seed: int


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
