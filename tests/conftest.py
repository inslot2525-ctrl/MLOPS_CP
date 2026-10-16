import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_30k():
    return make_synthetic(30_000, with_id=True)


@pytest.fixture(scope="session")
def synthetic_clean(synthetic_30k):
    return synthetic_30k.drop(columns=["ID"])
