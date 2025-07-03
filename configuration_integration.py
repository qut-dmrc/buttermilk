"""Configuration integration for enhanced Record with multi-field embedding support.

This module shows how to integrate the enhanced Record class with the existing
configuration system, including StorageConfig and multi-field embedding configurations.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json

from pydantic import BaseModel, Field, validator, root_validator
from omegaconf import DictConfig, OmegaConf

from enhanced_record_design import EnhancedRecord, ChunkData
from buttermilk._core.storage_config import StorageConfig, MultiFieldEmbeddingConfig, AdditionalFieldConfig
from buttermilk import logger

# ========== ENHANCED STORAGE CONFIGURATION ==========

class EnhancedStorageConfig(StorageConfig):
    """Enhanced StorageConfig with additional vector processing options."""
    
    # Enhanced Record specific fields
    enable_enhanced_records: bool = Field(
        default=True,
        description="Use EnhancedRecord format instead of legacy InputDocument"
    )
    
    auto_migrate_legacy: bool = Field(
        default=True,
        description="Automatically migrate legacy InputDocument to EnhancedRecord"
    )
    
    lazy_chunk_loading: bool = Field(
        default=False,
        description="Load chunks lazily from storage to save memory"
    )
    
    chunk_cache_size: int = Field(
        default=1000,
        description="Maximum number of chunk sets to keep in memory cache"
    )
    
    # Vector processing optimization
    embedding_cache_dir: Optional[str] = Field(
        default=None,
        description="Directory for caching embeddings to avoid recomputation"
    )
    
    parallel_chunk_processing: bool = Field(
        default=True,
        description="Process chunks in parallel for better performance"
    )
    
    max_chunk_workers: int = Field(
        default=4,
        description="Maximum number of parallel workers for chunk processing"
    )

    # Advanced multi-field configuration
    field_priority_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Priority weights for different field types during search (field_name: weight)"
    )
    
    dynamic_chunking: bool = Field(
        default=False,
        description="Enable dynamic chunk sizing based on content analysis"
    )
    
    cross_field_deduplication: bool = Field(
        default=True,
        description="Remove duplicate content across different fields"
    )

    @validator('chunk_cache_size')
    def validate_cache_size(cls, v):
        if v < 0:
            raise ValueError("chunk_cache_size must be non-negative")
        return v

    @validator('max_chunk_workers') 
    def validate_max_workers(cls, v):
        if v < 1:
            raise ValueError("max_chunk_workers must be at least 1")
        return v


# ========== ADVANCED MULTI-FIELD CONFIGURATION ==========

class ConditionalFieldConfig(BaseModel):
    """Configuration for conditional field embedding based on content analysis."""
    
    source_field: str = Field(description="Name of the field in Record.metadata")
    chunk_type: str = Field(description="Type tag for this chunk")
    min_length: int = Field(default=10, description="Minimum character length")
    max_length: Optional[int] = Field(default=None, description="Maximum character length")
    
    # Conditional embedding rules
    required_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns that must be present to embed this field"
    )
    
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns that prevent embedding this field"
    )
    
    content_quality_threshold: float = Field(
        default=0.0,
        description="Minimum content quality score (0.0-1.0) to embed this field"
    )
    
    language_codes: List[str] = Field(
        default_factory=list,
        description="Only embed if content language matches one of these codes"
    )


class DynamicChunkingConfig(BaseModel):
    """Configuration for dynamic chunking based on content analysis."""
    
    base_chunk_size: int = Field(default=1000, description="Base chunk size")
    size_variance: float = Field(default=0.2, description="Allowed size variance (0.0-1.0)")
    
    # Content-aware sizing
    sentence_boundary_priority: bool = Field(
        default=True,
        description="Prioritize sentence boundaries when chunking"
    )
    
    paragraph_boundary_priority: bool = Field(
        default=True,
        description="Prioritize paragraph boundaries when chunking"
    )
    
    semantic_boundary_detection: bool = Field(
        default=False,
        description="Use semantic analysis to detect natural chunk boundaries"
    )
    
    # Quality-based adjustments
    dense_content_smaller_chunks: bool = Field(
        default=True,
        description="Use smaller chunks for dense, technical content"
    )
    
    narrative_content_larger_chunks: bool = Field(
        default=True,
        description="Use larger chunks for narrative, flowing content"
    )


class AdvancedMultiFieldConfig(MultiFieldEmbeddingConfig):
    """Advanced multi-field embedding configuration with enhanced features."""
    
    # Replace simple additional_fields with conditional fields
    conditional_fields: List[ConditionalFieldConfig] = Field(
        default_factory=list,
        description="Advanced field configurations with conditional rules"
    )
    
    # Dynamic chunking
    dynamic_chunking_config: Optional[DynamicChunkingConfig] = Field(
        default=None,
        description="Configuration for dynamic chunking behavior"
    )
    
    # Cross-field processing
    field_relationships: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Define relationships between fields (field: [related_fields])"
    )
    
    combined_field_chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Create chunks by combining multiple fields"
    )
    
    # Quality control
    min_embedding_quality: float = Field(
        default=0.0,
        description="Minimum quality score for embeddings (0.0-1.0)"
    )
    
    quality_metrics: List[str] = Field(
        default_factory=lambda: ["length", "uniqueness", "informativeness"],
        description="Quality metrics to evaluate for chunk filtering"
    )

    @root_validator
    def validate_field_configs(cls, values):
        """Validate that field configurations are consistent."""
        conditional_fields = values.get('conditional_fields', [])
        additional_fields = values.get('additional_fields', [])
        
        if conditional_fields and additional_fields:
            logger.warning("Both conditional_fields and additional_fields specified. Using conditional_fields.")
            values['additional_fields'] = []
        
        return values


# ========== CONFIGURATION FACTORY ==========

class EnhancedConfigurationFactory:
    """Factory for creating enhanced configurations from various sources."""
    
    @staticmethod
    def from_hydra_config(cfg: DictConfig) -> EnhancedStorageConfig:
        """Create EnhancedStorageConfig from Hydra configuration."""
        
        # Convert OmegaConf to dict
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Handle multi-field embedding configuration
        if 'multi_field_embedding' in config_dict:
            multi_field_data = config_dict['multi_field_embedding']
            
            # Convert to AdvancedMultiFieldConfig if it has advanced features
            if 'conditional_fields' in multi_field_data or 'dynamic_chunking_config' in multi_field_data:
                config_dict['multi_field_embedding'] = AdvancedMultiFieldConfig(**multi_field_data)
            else:
                config_dict['multi_field_embedding'] = MultiFieldEmbeddingConfig(**multi_field_data)
        
        return EnhancedStorageConfig(**config_dict)
    
    @staticmethod
    def from_yaml_file(file_path: Path) -> EnhancedStorageConfig:
        """Create configuration from YAML file."""
        cfg = OmegaConf.load(file_path)
        return EnhancedConfigurationFactory.from_hydra_config(cfg)
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> EnhancedStorageConfig:
        """Create configuration from dictionary."""
        cfg = OmegaConf.create(config_dict)
        return EnhancedConfigurationFactory.from_hydra_config(cfg)


# ========== CONFIGURATION VALIDATION ==========

class ConfigurationValidator:
    """Validates enhanced configurations for consistency and correctness."""
    
    @staticmethod
    def validate_multi_field_config(config: AdvancedMultiFieldConfig) -> List[str]:
        """Validate multi-field configuration and return list of issues."""
        issues = []
        
        # Check chunk size settings
        if config.chunk_size <= 0:
            issues.append("chunk_size must be positive")
        
        if config.chunk_overlap >= config.chunk_size:
            issues.append("chunk_overlap must be less than chunk_size")
        
        # Validate conditional fields
        for field_config in config.conditional_fields:
            if field_config.min_length < 0:
                issues.append(f"min_length for field '{field_config.source_field}' must be non-negative")
            
            if field_config.max_length is not None and field_config.max_length < field_config.min_length:
                issues.append(f"max_length for field '{field_config.source_field}' must be >= min_length")
        
        # Validate quality thresholds
        if not 0.0 <= config.min_embedding_quality <= 1.0:
            issues.append("min_embedding_quality must be between 0.0 and 1.0")
        
        return issues
    
    @staticmethod
    def validate_storage_config(config: EnhancedStorageConfig) -> List[str]:
        """Validate storage configuration and return list of issues."""
        issues = []
        
        # Basic validation
        if config.type not in ["bigquery", "file", "gcs", "s3", "local", "chromadb", "vector"]:
            issues.append(f"Unsupported storage type: {config.type}")
        
        # Vector-specific validation
        if config.type in ["chromadb", "vector"]:
            if not config.collection_name:
                issues.append("collection_name is required for vector storage")
            
            if config.dimensionality and config.dimensionality <= 0:
                issues.append("dimensionality must be positive")
        
        # Multi-field config validation
        if hasattr(config, 'multi_field_embedding') and config.multi_field_embedding:
            if isinstance(config.multi_field_embedding, AdvancedMultiFieldConfig):
                multi_field_issues = ConfigurationValidator.validate_multi_field_config(
                    config.multi_field_embedding
                )
                issues.extend(multi_field_issues)
        
        return issues


# ========== CONFIGURATION TEMPLATES ==========

class ConfigurationTemplates:
    """Pre-defined configuration templates for common use cases."""
    
    @staticmethod
    def academic_papers() -> AdvancedMultiFieldConfig:
        """Configuration template for academic papers."""
        return AdvancedMultiFieldConfig(
            content_field="content",
            conditional_fields=[
                ConditionalFieldConfig(
                    source_field="title",
                    chunk_type="title",
                    min_length=5,
                    max_length=500
                ),
                ConditionalFieldConfig(
                    source_field="abstract",
                    chunk_type="abstract", 
                    min_length=50,
                    max_length=5000
                ),
                ConditionalFieldConfig(
                    source_field="keywords",
                    chunk_type="keywords",
                    min_length=5,
                    max_length=1000
                ),
                ConditionalFieldConfig(
                    source_field="conclusion",
                    chunk_type="conclusion",
                    min_length=100,
                    content_quality_threshold=0.3
                )
            ],
            chunk_size=1000,
            chunk_overlap=200,
            dynamic_chunking_config=DynamicChunkingConfig(
                base_chunk_size=1000,
                semantic_boundary_detection=True,
                dense_content_smaller_chunks=True
            ),
            min_embedding_quality=0.2
        )
    
    @staticmethod
    def news_articles() -> AdvancedMultiFieldConfig:
        """Configuration template for news articles."""
        return AdvancedMultiFieldConfig(
            content_field="content",
            conditional_fields=[
                ConditionalFieldConfig(
                    source_field="headline",
                    chunk_type="headline",
                    min_length=10,
                    max_length=200
                ),
                ConditionalFieldConfig(
                    source_field="summary",
                    chunk_type="summary",
                    min_length=50,
                    max_length=1000
                ),
                ConditionalFieldConfig(
                    source_field="byline",
                    chunk_type="byline",
                    min_length=5,
                    max_length=100
                )
            ],
            chunk_size=800,
            chunk_overlap=150,
            dynamic_chunking_config=DynamicChunkingConfig(
                base_chunk_size=800,
                paragraph_boundary_priority=True,
                narrative_content_larger_chunks=True
            )
        )
    
    @staticmethod
    def technical_documentation() -> AdvancedMultiFieldConfig:
        """Configuration template for technical documentation."""
        return AdvancedMultiFieldConfig(
            content_field="content",
            conditional_fields=[
                ConditionalFieldConfig(
                    source_field="title",
                    chunk_type="title",
                    min_length=5,
                    max_length=200
                ),
                ConditionalFieldConfig(
                    source_field="api_reference",
                    chunk_type="api_reference",
                    min_length=20,
                    required_patterns=[r"def\s+\w+", r"class\s+\w+", r"function\s+\w+"]
                ),
                ConditionalFieldConfig(
                    source_field="code_examples",
                    chunk_type="code_example",
                    min_length=30,
                    required_patterns=[r"```", r"<code>", r"def\s+", r"class\s+"]
                )
            ],
            chunk_size=600,
            chunk_overlap=100,
            dynamic_chunking_config=DynamicChunkingConfig(
                base_chunk_size=600,
                dense_content_smaller_chunks=True,
                sentence_boundary_priority=False  # Code doesn't follow sentence structure
            )
        )


# ========== CONFIGURATION MIGRATION ==========

class ConfigurationMigrator:
    """Migrates legacy configurations to enhanced format."""
    
    @staticmethod
    def migrate_legacy_multi_field_config(
        legacy_config: MultiFieldEmbeddingConfig
    ) -> AdvancedMultiFieldConfig:
        """Migrate legacy MultiFieldEmbeddingConfig to AdvancedMultiFieldConfig."""
        
        # Convert additional_fields to conditional_fields
        conditional_fields = []
        for field_config in legacy_config.additional_fields:
            conditional_field = ConditionalFieldConfig(
                source_field=field_config.source_field,
                chunk_type=field_config.chunk_type,
                min_length=field_config.min_length
            )
            conditional_fields.append(conditional_field)
        
        return AdvancedMultiFieldConfig(
            content_field=legacy_config.content_field,
            conditional_fields=conditional_fields,
            chunk_size=legacy_config.chunk_size,
            chunk_overlap=legacy_config.chunk_overlap,
            # Default empty values for new fields
            additional_fields=[]  # Clear legacy field
        )
    
    @staticmethod
    def migrate_storage_config(legacy_config: StorageConfig) -> EnhancedStorageConfig:
        """Migrate legacy StorageConfig to EnhancedStorageConfig."""
        
        # Extract existing data
        config_data = legacy_config.model_dump()
        
        # Add default values for new fields
        enhanced_defaults = {
            'enable_enhanced_records': True,
            'auto_migrate_legacy': True,
            'lazy_chunk_loading': False,
            'chunk_cache_size': 1000,
            'parallel_chunk_processing': True,
            'max_chunk_workers': 4,
            'field_priority_weights': {},
            'dynamic_chunking': False,
            'cross_field_deduplication': True
        }
        
        # Merge with defaults
        config_data.update(enhanced_defaults)
        
        # Migrate multi-field config if present
        if config_data.get('multi_field_embedding'):
            multi_field_data = config_data['multi_field_embedding']
            if isinstance(multi_field_data, dict):
                # Convert to AdvancedMultiFieldConfig
                if 'additional_fields' in multi_field_data:
                    legacy_multi_field = MultiFieldEmbeddingConfig(**multi_field_data)
                    config_data['multi_field_embedding'] = ConfigurationMigrator.migrate_legacy_multi_field_config(
                        legacy_multi_field
                    )
        
        return EnhancedStorageConfig(**config_data)


# ========== CONFIGURATION UTILITIES ==========

class ConfigurationUtils:
    """Utility functions for working with enhanced configurations."""
    
    @staticmethod
    def get_field_embedding_strategy(
        config: AdvancedMultiFieldConfig,
        field_name: str,
        content: str
    ) -> Dict[str, Any]:
        """Determine embedding strategy for a specific field and content."""
        
        strategy = {
            'should_embed': False,
            'chunk_type': 'unknown',
            'chunk_size': config.chunk_size,
            'quality_threshold': config.min_embedding_quality,
            'reasons': []
        }
        
        # Find matching conditional field config
        field_config = None
        for cond_field in config.conditional_fields:
            if cond_field.source_field == field_name:
                field_config = cond_field
                break
        
        if not field_config:
            strategy['reasons'].append('No configuration found for field')
            return strategy
        
        # Check length requirements
        content_length = len(content.strip())
        if content_length < field_config.min_length:
            strategy['reasons'].append(f'Content too short ({content_length} < {field_config.min_length})')
            return strategy
        
        if field_config.max_length and content_length > field_config.max_length:
            strategy['reasons'].append(f'Content too long ({content_length} > {field_config.max_length})')
            return strategy
        
        # Check pattern requirements
        import re
        for pattern in field_config.required_patterns:
            if not re.search(pattern, content):
                strategy['reasons'].append(f'Missing required pattern: {pattern}')
                return strategy
        
        for pattern in field_config.exclude_patterns:
            if re.search(pattern, content):
                strategy['reasons'].append(f'Contains excluded pattern: {pattern}')
                return strategy
        
        # If we get here, should embed
        strategy.update({
            'should_embed': True,
            'chunk_type': field_config.chunk_type,
            'quality_threshold': field_config.content_quality_threshold,
            'reasons': ['All conditions met']
        })
        
        return strategy
    
    @staticmethod
    def optimize_chunk_size(
        content: str,
        base_size: int,
        dynamic_config: Optional[DynamicChunkingConfig] = None
    ) -> int:
        """Optimize chunk size based on content characteristics."""
        
        if not dynamic_config:
            return base_size
        
        # Analyze content characteristics
        sentences = content.count('.') + content.count('!') + content.count('?')
        paragraphs = content.count('\n\n') + 1
        avg_sentence_length = len(content) / max(sentences, 1)
        
        # Adjust size based on content density
        adjusted_size = base_size
        
        if dynamic_config.dense_content_smaller_chunks:
            # Dense content (short sentences, many technical terms)
            if avg_sentence_length < 50:
                adjusted_size = int(base_size * 0.8)
        
        if dynamic_config.narrative_content_larger_chunks:
            # Narrative content (long sentences, flowing text)
            if avg_sentence_length > 100:
                adjusted_size = int(base_size * 1.2)
        
        # Apply variance limits
        min_size = int(base_size * (1 - dynamic_config.size_variance))
        max_size = int(base_size * (1 + dynamic_config.size_variance))
        
        return max(min_size, min(max_size, adjusted_size))


# ========== EXAMPLE USAGE ==========

def example_configuration_usage():
    """Example of how to use the enhanced configuration system."""
    
    # 1. Create configuration from template
    config = ConfigurationTemplates.academic_papers()
    
    # 2. Validate configuration
    issues = ConfigurationValidator.validate_multi_field_config(config)
    if issues:
        print(f"Configuration issues: {issues}")
        return
    
    # 3. Create enhanced storage config
    storage_config = EnhancedStorageConfig(
        type="chromadb",
        collection_name="academic_papers",
        persist_directory="./chromadb",
        multi_field_embedding=config,
        enable_enhanced_records=True,
        dynamic_chunking=True
    )
    
    # 4. Test field embedding strategy
    test_content = "This is a comprehensive study of machine learning algorithms and their applications."
    strategy = ConfigurationUtils.get_field_embedding_strategy(
        config, "abstract", test_content
    )
    
    print(f"Embedding strategy: {strategy}")
    
    # 5. Optimize chunk size
    optimized_size = ConfigurationUtils.optimize_chunk_size(
        test_content, config.chunk_size, config.dynamic_chunking_config
    )
    
    print(f"Optimized chunk size: {optimized_size}")


if __name__ == "__main__":
    example_configuration_usage()