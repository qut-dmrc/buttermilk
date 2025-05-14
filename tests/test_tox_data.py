import pytest

from buttermilk._core.types import Record

# import evaluate
pytest.importorskip("torch")


def toxic_record() -> Record:
    from buttermilk.toxicity.datasets import ImplicitHatePipe
    datasource = ImplicitHatePipe()
    rec = next(datasource)

    return rec
