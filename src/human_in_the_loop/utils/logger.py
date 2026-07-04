"""Structured logging utility."""
import sys
from pathlib import Path
import structlog
from human_in_the_loop.config.settings import get_settings

def get_logger(module_name: str) -> structlog.BoundLogger:
    settings = get_settings()
    Path(settings.logging.file).parent.mkdir(parents=True, exist_ok=True)
    renderer = structlog.processors.JSONRenderer() if settings.logging.format == "json" else structlog.dev.ConsoleRenderer()
    level_map = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
    structlog.configure(
        processors=[structlog.contextvars.merge_contextvars, structlog.processors.add_log_level, structlog.processors.TimeStamper(fmt="iso"), structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level_map.get(settings.logging.level.upper(), 20)),
        context_class=dict, logger_factory=structlog.PrintLoggerFactory(file=sys.stdout), cache_logger_on_first_use=True,
    )
    return structlog.get_logger(module=module_name)
