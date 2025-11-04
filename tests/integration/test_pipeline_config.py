from pathlib import Path

from omegaconf import OmegaConf


def test_image_generation_config_file_exists():
    """Test that image generation pipeline config file exists and is valid YAML."""
    config_path = Path(__file__).parent.parent.parent / "buttermilk" / "conf" / "pipelines" / "image_generation.yaml"

    assert config_path.exists(), f"Config file not found at {config_path}"

    # Load and parse YAML
    cfg = OmegaConf.load(config_path)

    assert "pipeline" in cfg
    assert cfg.pipeline.name == "image_generation"
    assert "source" in cfg.pipeline
    assert "processors" in cfg.pipeline


def test_image_generation_source_config():
    """Test pipeline source configuration is correct."""
    config_path = Path(__file__).parent.parent.parent / "buttermilk" / "conf" / "pipelines" / "image_generation.yaml"
    cfg = OmegaConf.load(config_path)

    # Verify source configuration
    source = cfg.pipeline.source
    assert source._target_ == "buttermilk.data.sources.character_prompt_source.CharacterPromptSource"
    assert "sexuality" in source.mask_attributes
    assert "gender" in source.mask_attributes
    assert len(source.scenarios) == 5  # Default has 5 scenarios


def test_image_generation_processors_config():
    """Test pipeline processors are configured correctly."""
    config_path = Path(__file__).parent.parent.parent / "buttermilk" / "conf" / "pipelines" / "image_generation.yaml"
    cfg = OmegaConf.load(config_path)

    processors = cfg.pipeline.processors

    # Verify processor types and count
    assert len(processors) == 4
    assert processors[0]._target_ == "buttermilk.processors.batch_expansion.BatchExpansionProcessor"
    assert processors[1]._target_ == "buttermilk.processors.image_generation.ImageGenerationProcessor"
    assert processors[2]._target_ == "buttermilk.processors.gcs_image_storage.GCSImageStorageProcessor"
    assert processors[3]._target_ == "buttermilk.processors.csv_metadata_logger.CSVMetadataLogger"

    # Verify batch expansion config
    assert processors[0].repetitions == 3
    assert len(processors[0].models) == 2


def test_image_generation_testing_override():
    """Test that testing.yaml contains overrides for image generation pipeline."""
    testing_config_path = Path(__file__).parent.parent.parent / "buttermilk" / "conf" / "testing.yaml"
    cfg = OmegaConf.load(testing_config_path)

    # Verify testing overrides exist for image_generation pipeline
    assert "pipelines" in cfg
    assert "image_generation" in cfg.pipelines

    # Verify testing reduces scenarios (2 instead of 5)
    assert len(cfg.pipelines.image_generation.source.scenarios) == 2
    assert "working in an office" in cfg.pipelines.image_generation.source.scenarios
    assert "at a coffee shop" in cfg.pipelines.image_generation.source.scenarios

    # Verify testing reduces repetitions (2 instead of 3)
    assert cfg.pipelines.image_generation.processors[0].repetitions == 2

    # Verify testing uses only one model (VertexImagen3Fast)
    assert len(cfg.pipelines.image_generation.processors[0].models) == 1
