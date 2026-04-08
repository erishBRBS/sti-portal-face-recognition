from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "face-recognition-api"
    app_host: str = "0.0.0.0"
    app_port: int = 9000

    face_model_name: str = "buffalo_l"
    face_similarity_threshold: float = 0.55
    face_det_width: int = 640
    face_det_height: int = 640

    embeddings_file: str = "/app/data/embeddings.json"

    laravel_api_base_url: str = "http://laravel-api:8001/api"
    laravel_api_token: str = ""
    request_timeout_seconds: int = 30
    notify_laravel: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()