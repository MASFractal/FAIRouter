import pytest

from fai_router.settings import Settings


@pytest.fixture(autouse=True)
def reset_settings():
    """Настройки статические, поэтому каждый тест начинает с чистого состояния."""
    Settings.task_mean = None
    Settings.llm = None
    Settings.temperature_scale = 20.0
    Settings.WQ, Settings.WC, Settings.WT = 0.5, 0.25, 0.25
    yield
    Settings.task_mean = None
    Settings.llm = None
