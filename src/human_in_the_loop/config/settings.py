"""Settings for human-in-the-loop framework."""
import os
from functools import lru_cache
from pathlib import Path
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class ApprovalSettings(BaseSettings):
    confidence_threshold: float = Field(default=0.8)
    auto_approve_above: float = Field(default=0.95)
    auto_reject_below: float = Field(default=0.2)
    timeout_seconds: int = Field(default=3600)
    escalation_after_seconds: int = Field(default=1800)


class QueueSettings(BaseSettings):
    max_pending_items: int = Field(default=1000)
    priority_enabled: bool = Field(default=True)
    assignment_strategy: str = Field(default="round_robin")


class LearningSettings(BaseSettings):
    enabled: bool = Field(default=True)
    min_feedback_samples: int = Field(default=10)
    retrain_threshold: int = Field(default=50)


class AuditSettings(BaseSettings):
    enabled: bool = Field(default=True)
    retention_days: int = Field(default=90)


class APISettings(BaseSettings):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)


class LoggingSettings(BaseSettings):
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file: str = Field(default="logs/app.log")


class Settings(BaseSettings):
    approval: ApprovalSettings = Field(default_factory=ApprovalSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    learning: LearningSettings = Field(default_factory=LearningSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    model_config = {"env_prefix": "", "env_nested_delimiter": "__"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env = os.getenv("APP_ENV", "development")
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    env_map = {"development": "dev", "production": "prod"}
    suffix = env_map.get(env, "")
    config_file = config_dir / f"application-{suffix}.yaml" if suffix else config_dir / "application.yaml"
    if not config_file.exists():
        config_file = config_dir / "application.yaml"
    cfg = {}
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f) or {}
    return Settings(
        approval=ApprovalSettings(**cfg.get("approval", {})) if cfg.get("approval") else ApprovalSettings(),
        queue=QueueSettings(**cfg.get("queue", {})) if cfg.get("queue") else QueueSettings(),
        learning=LearningSettings(**cfg.get("learning", {})) if cfg.get("learning") else LearningSettings(),
        audit=AuditSettings(**cfg.get("audit", {})) if cfg.get("audit") else AuditSettings(),
        api=APISettings(**cfg.get("api", {})) if cfg.get("api") else APISettings(),
        logging=LoggingSettings(**cfg.get("logging", {})) if cfg.get("logging") else LoggingSettings(),
    )
