
from cloudpathlib import AnyPath


def test_save(real_bm):
    uri = real_bm.save(data=["test data"], extension=".txt")
    assert uri.startswith(real_bm.session_info.save_dir)
    assert uri.endswith(".txt")
    uploaded = AnyPath(uri)
    assert uploaded.exists()
    read_text = uploaded.read_text()
    # The data is saved as a JSON-serialized list, not a single string
    assert read_text == '["test data"]'
    uploaded.unlink(missing_ok=False)

