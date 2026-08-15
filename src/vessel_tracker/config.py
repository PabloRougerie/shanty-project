"""Project configuration loaded from configs/default.yaml."""

from pydantic import BaseModel, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource

from vessel_tracker.paths import CONFIGS_DIR


class HorizonConfig(BaseModel):
    """Named prediction horizon in minutes (v3 grid)."""

    name: str
    minutes: int


class SplitRatios(BaseModel):
    train: float
    val: float
    test: float


class BoundingBox(BaseModel):
    lon_west: float
    lon_east: float
    lat_south: float
    lat_north: float


class IngestionConfig(BaseModel):
    """Paramètres de téléchargement des données AIS depuis NOAA."""

    start_date: str
    end_date: str
    bounding_box: BoundingBox


class Settings(BaseSettings):
    random_seed: int = 273
    resample_interval_min: int = 10
    split_ratios: SplitRatios
    min_track_steps: int = 504
    lookback_minutes: int = 720
    nb_lags: int = 6
    feature_columns: list[str]
    horizons: list[HorizonConfig]
    ingestion: IngestionConfig

    model_config = SettingsConfigDict(
        yaml_file=CONFIGS_DIR / "default.yaml",
        yaml_file_encoding="utf-8",
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (YamlConfigSettingsSource(settings_cls),)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def horizon_grid_minutes(self) -> list[int]:
        """Horizon grid in minutes, ordered as in default.yaml (NB07)."""
        return [h.minutes for h in self.horizons]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_horizon_minutes(self) -> int:
        return max(self.horizon_grid_minutes)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_horizon_hours(self) -> int:
        return self.max_horizon_minutes // 60

    def horizon_steps(self, horizon_minutes: int) -> int:
        """Nombre de pas de temps pour un horizon donné."""
        return horizon_minutes // self.resample_interval_min

    def lookback_steps(self, lookback_minutes: int | None = None) -> int:
        """Nombre de pas de temps pour le lookback (défaut : lookback_minutes du YAML)."""
        minutes = self.lookback_minutes if lookback_minutes is None else lookback_minutes
        return minutes // self.resample_interval_min


settings = Settings()
