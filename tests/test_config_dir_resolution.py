import importlib
import types

# Import module under test at top-level to satisfy linters
cb = importlib.import_module("buttermilk._core.config_bootstrap")


def test_relative_config_dir_resolves_against_cwd(monkeypatch, tmp_path):
    """Ensure relative config_dir is resolved relative to caller's CWD."""
    # Arrange

    # Switch CWD to a temp project directory
    project_dir = tmp_path / "proj"
    conf_dir = project_dir / "myconf"
    conf_dir.mkdir(parents=True)
    monkeypatch.chdir(project_dir)

    captured = {}

    class DummyBootstrapper:
        def __init__(self, *, config_path: str, overrides=None, config=None):
            captured["config_path"] = config_path

        @staticmethod
        async def bootstrap_full_context():
            return Dummy()

        @staticmethod
        async def bootstrap_session_context(name: str, job: str, **kwargs):
            return Dummy()

        @staticmethod
        def get_configuration():
            session_info = types.SimpleNamespace(job="cfgjob", name="cfgproj")
            bm = types.SimpleNamespace(session_info=session_info)
            return types.SimpleNamespace(bm=bm)

    class Dummy:
        class _SI:
            project_name = "proj"
            job = "job"

        session_info = _SI()

        @staticmethod
        def validate_and_set_project(name):
            return name

    # Make set_bm a no-op and replace the bootstrapper class
    monkeypatch.setattr(cb, "ConfigurationBootstrapper", DummyBootstrapper)
    # Patch import target for set_bm, since function imports it inside
    monkeypatch.setattr("buttermilk._core.config_bootstrap.set_bm", lambda _: None, raising=False)

    # Replace asyncio.run globally to avoid running async code
    monkeypatch.setattr("asyncio.run", lambda _: Dummy(), raising=False)

    # Act
    cb.bootstrap_session_with_config(
        job="j",
        project="p",
        run_type="cli",
        config_dir="myconf",  # relative path should resolve against project_dir (cwd)
        overrides=None,
        config=None,
    )

    # Assert
    expected = conf_dir.resolve().as_posix()
    assert captured.get("config_path") == expected


def test_tilde_and_env_expansion(monkeypatch, tmp_path):
    """Ensure ~ and $VARS are expanded and resolved."""

    # Create a fake HOME and set CWD elsewhere
    fake_home = tmp_path / "homeuser"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    project_dir = tmp_path / "another_proj"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    # Create dir under ~ and refer via env var
    home_conf = fake_home / "bmconf"
    home_conf.mkdir()
    monkeypatch.setenv("BM_CONF", "bmconf")

    captured = {}

    class DummyBootstrapper:
        def __init__(self, *, config_path: str, overrides=None, config=None):
            captured["config_path"] = config_path

        @staticmethod
        async def bootstrap_full_context():
            return Dummy()

        @staticmethod
        async def bootstrap_session_context(name: str, job: str, **kwargs):
            return Dummy()

        @staticmethod
        def get_configuration():
            session_info = types.SimpleNamespace(job="cfgjob", name="cfgproj")
            bm = types.SimpleNamespace(session_info=session_info)
            return types.SimpleNamespace(bm=bm)

    class Dummy:
        class _SI:
            project_name = "proj"
            job = "job"

        session_info = _SI()

        @staticmethod
        def validate_and_set_project(name):
            return name

    monkeypatch.setattr(cb, "ConfigurationBootstrapper", DummyBootstrapper)
    monkeypatch.setattr("buttermilk._core.config_bootstrap.set_bm", lambda _: None, raising=False)
    monkeypatch.setattr("asyncio.run", lambda _: Dummy(), raising=False)

    # ~ expansion
    cb.bootstrap_session_with_config(config_dir="~/bmconf")
    assert captured.get("config_path") == home_conf.resolve().as_posix()

    # $VARS expansion
    cb.bootstrap_session_with_config(config_dir="$HOME/$BM_CONF")
    assert captured.get("config_path") == home_conf.resolve().as_posix()
