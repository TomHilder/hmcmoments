# settings.py
# Writen by Thomas Hilder

"""
Settings dataclass to store settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

from .models import MODEL_NAMES


@dataclass(frozen=True)
class Settings:
    # Default values for settings
    DEFAULT_MODEL = 1
    DEFAULT_DOWNSAMPLE = None
    DEFAULT_CORES = cpu_count()
    DEFAULT_OVERWRITE = False
    # Attributes
    file: Path
    model: int
    downsample: int
    cores: int
    overwrite: bool

    # Constructor
    @classmethod
    def from_dict(
        cls,
        filename: str,
        model: str = DEFAULT_MODEL,
        downsample: int = DEFAULT_DOWNSAMPLE,
        cores: int = DEFAULT_CORES,
        overwrite: bool = DEFAULT_OVERWRITE,
    ) -> Settings:
        file = Path(filename)
        if not file.exists():
            raise FileNotFoundError(f"Could not find file {filename}.")
        return cls(file, model, downsample, cores, overwrite)

    # Base filename for outputs
    @property
    def output_fname_base(self) -> str:
        return f"{MODEL_NAMES[self.model]}_ds{self.downsample}_{self.file.name}"
