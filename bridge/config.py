"""Bridge configuration — loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Bridge Band agent key (band_a_...)
    bridge_agent_key: str

    # Registered Band agent IDs (obtained after register_my_agent)
    triage_agent_id: str
    knowledge_agent_id: str
    compliance_agent_id: str

    # Band API
    band_base_url: str = "https://app.band.ai"

    # Renggo outbound call endpoint (for callback path)
    renggo_outbound_url: str = "http://localhost:8000/api/v1/calls/outbound"
    renggo_api_key: str = ""

    # FastAPI
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    # Timeouts
    in_call_timeout_s: int = 20    # max wait for in-call path resolution


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
