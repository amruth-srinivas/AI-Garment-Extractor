import os
import tempfile
from pathlib import Path

_root = Path(tempfile.mkdtemp(prefix="garmentai-test-"))
os.environ["GARMENTAI_DATA_DIR"] = str(_root / "data")
os.environ["GARMENTAI_WEIGHTS_DIR"] = str(_root / "weights")
os.environ["GARMENTAI_HF_HOME"] = str(_root / "hf")
os.environ["GARMENTAI_AUTO_DOWNLOAD"] = "0"
os.environ["HF_HOME"] = str(_root / "hf")

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client
