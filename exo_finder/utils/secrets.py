from pydantic_settings import BaseSettings, SettingsConfigDict

from paths import PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )

    MAST_TOKEN: str
    GAIA_USER: str
    GAIA_PASSWORD: str
    CASJOB_WSID: int
    CASJOB_PASSWORD: str


settings = Settings()
