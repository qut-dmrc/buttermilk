from pathlib import Path

from buttermilk._core.storage_config import BigQueryStorageConfig
from buttermilk.storage.bigquery import BigQueryStorage


class FakeQueryJob:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        yield from self._rows


class FakeRow:
    def __init__(self, data: dict):
        self._data = data

    def items(self):
        return self._data.items()


class FakeClient:
    def __init__(self):
        self.last_query = None
        self.last_params = None

    def query(self, query: str, job_config):
        self.last_query = query
        # BigQuery QueryJobConfig has attribute query_parameters
        self.last_params = getattr(job_config, "query_parameters", None)
        # Return a single row by default; tests can monkeypatch this behavior
        record = {
            "record_id": "rec-1",
            "content": "hello",
            "metadata": {"k": "v"},
            "ground_truth": {"g": 1},
            "mime": "text/plain",
            "dataset_name": "ds",
            "split_type": "train",
        }
        return FakeQueryJob([FakeRow(record)])


def make_config(**overrides) -> BigQueryStorageConfig:
    root = Path(__file__).resolve().parents[1]
    defaults = dict(
        type="bigquery",
        project_id="proj",
        dataset_id="ds",
        table_id="tbl",
        dataset_name="my_dataset",
        schema_path=str(root / "buttermilk" / "schemas" / "record.schema.json"),
        split_type="train",
        columns={},
    )
    defaults.update(overrides)
    return BigQueryStorageConfig(**defaults)


def make_storage(config: BigQueryStorageConfig) -> BigQueryStorage:
    storage = BigQueryStorage(config)
    # Inject fake client to avoid real BQ calls
    storage._client = FakeClient()
    return storage


def test_get_record_by_id_happy_path(monkeypatch):
    cfg = make_config()
    storage = make_storage(cfg)

    # Pretend table has the expected logical columns
    monkeypatch.setattr(
        storage,
        "_available_columns",
        lambda: {"record_id", "dataset_name", "split_type"},
    )

    rec = storage.get_record_by_id("rec-1")
    assert rec is not None
    assert rec.record_id == "rec-1"
    assert rec.content == "hello"
    assert rec.metadata["k"] == "v"

    # Verify query parameters include all three
    params = {p.name for p in storage.client.last_params}
    assert params == {"record_id", "dataset_name", "split_type"}


def test_get_record_by_id_skips_missing_filters(monkeypatch):
    cfg = make_config(split_type="train")
    storage = make_storage(cfg)

    # Only record_id exists; dataset and split do not
    monkeypatch.setattr(storage, "_available_columns", lambda: {"record_id"})

    # Capture query with a custom FakeClient that records query string
    fc = FakeClient()
    storage._client = fc

    rec = storage.get_record_by_id("rec-1")
    assert rec is not None

    # Ensure dataset/split filters are not present in SQL
    q = fc.last_query
    assert "record_id = @record_id" in q
    assert "dataset_name = @dataset_name" not in q
    assert "split_type = @split_type" not in q

    # But parameters may still include only record_id
    param_names = {p.name for p in fc.last_params}
    assert param_names == {"record_id"}


def test_get_record_by_id_respects_column_mapping(monkeypatch):
<<<<<<< HEAD
    cfg = make_config(columns={"record_id": "id", "dataset_name": "dataset", "split_type": "fold"})
    storage = make_storage(cfg)

    # Available columns reflect the mapped names
    monkeypatch.setattr(storage, "_available_columns", lambda: {"id", "dataset", "fold"})
=======
    cfg = make_config(
        columns={"record_id": "id", "dataset_name": "dataset", "split_type": "fold"}
    )
    storage = make_storage(cfg)

    # Available columns reflect the mapped names
    monkeypatch.setattr(
        storage, "_available_columns", lambda: {"id", "dataset", "fold"}
    )
>>>>>>> origin/stable

    fc = FakeClient()
    storage._client = fc

    _ = storage.get_record_by_id("xyz")

    q = fc.last_query
    assert "id = @record_id" in q
    assert "dataset = @dataset_name" in q
    assert "fold = @split_type" in q

    # Parameter names remain logical and should be present
    param_names = {p.name for p in fc.last_params}
    assert {"record_id", "dataset_name", "split_type"}.issubset(param_names)
