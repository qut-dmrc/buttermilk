"""
Configuration Validator for Runtime Debugging

Flow-agnostic configuration validation system for buttermilk debugging.
Provides detailed validation of configuration files, type schemas, and dependencies.
"""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from buttermilk._core.log import logger


class ValidationIssue(BaseModel):
    """Represents a configuration validation issue."""
    
    severity: str = Field(description="Severity level: error, warning, info")
    component: str = Field(description="Component where issue was found")
    field: str = Field(description="Specific field with issue")
    message: str = Field(description="Description of the issue")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix")
    file_path: Optional[str] = Field(default=None, description="File path where issue occurred")


class ValidationReport(BaseModel):
    """Complete validation report for configuration."""
    
    total_files_checked: int = Field(description="Number of files validated")
    errors: List[ValidationIssue] = Field(default_factory=list, description="Error-level issues")
    warnings: List[ValidationIssue] = Field(default_factory=list, description="Warning-level issues")
    info: List[ValidationIssue] = Field(default_factory=list, description="Info-level issues")
    passed_checks: List[str] = Field(default_factory=list, description="Successful validations")
    dependency_issues: List[str] = Field(default_factory=list, description="Missing dependencies")
    
    @property
    def total_issues(self) -> int:
        """Total number of validation issues."""
        return len(self.errors) + len(self.warnings) + len(self.info)
    
    @property
    def is_valid(self) -> bool:
        """True if no errors found."""
        return len(self.errors) == 0


class ConfigValidator:
    """
    Flow-agnostic configuration validator for buttermilk debugging.
    
    Validates configuration files, storage schemas, agent configurations,
    and dependency availability.
    """
    
    def __init__(self, base_config_path: str = "/workspaces/buttermilk/conf"):
        """
        Initialize configuration validator.
        
        Args:
            base_config_path: Path to configuration directory
        """
        self.base_path = Path(base_config_path)
        self.report = ValidationReport(total_files_checked=0)
    
    def validate_all(self) -> ValidationReport:
        """
        Run comprehensive validation of all configuration components.
        
        Returns:
            Complete validation report
        """
        logger.info("Starting comprehensive configuration validation...")
        
        # Reset report
        self.report = ValidationReport(total_files_checked=0)
        
        # Check dependencies first
        self._check_dependencies()
        
        # Validate configuration files
        self._validate_config_files()
        
        # Validate storage configurations
        self._validate_storage_configs()
        
        # Validate flow configurations
        self._validate_flow_configs()
        
        # Generate summary
        self._generate_summary()
        
        return self.report
    
    def _check_dependencies(self):
        """Check for optional dependencies that might cause import issues."""
        dependencies = [
            ("google.cloud.bigquery", "BigQuery operations"),
            ("google.cloud.storage", "Cloud Storage operations"),
            ("google.cloud.logging", "Cloud Logging"),
            ("autogen_core", "Autogen message types"),
            ("pdfminer", "PDF text extraction"),
            ("azure.identity", "Azure authentication"),
            ("chromadb", "Vector database operations"),
        ]
        
        for module_name, description in dependencies:
            try:
                __import__(module_name)
                self.report.passed_checks.append(f"Dependency available: {module_name}")
            except ImportError as e:
                self.report.dependency_issues.append(f"Optional dependency missing: {module_name} ({description})")
                self.report.warnings.append(ValidationIssue(
                    severity="warning",
                    component="dependencies",
                    field=module_name,
                    message=f"Optional dependency not available: {module_name}",
                    suggestion=f"Install {module_name} if you need {description}"
                ))
    
    def _validate_config_files(self):
        """Validate main configuration files."""
        config_files = [
            "config.yaml",
            "llm_defaults.yaml",
            "flows/osb.yaml",
            "flows/trans.yaml", 
            "flows/tox_allinone.yaml"
        ]
        
        for config_file in config_files:
            file_path = self.base_path / config_file
            self._validate_yaml_file(file_path, "config")
    
    def _validate_storage_configs(self):
        """Validate storage configuration files."""
        storage_dir = self.base_path / "storage"
        if not storage_dir.exists():
            self.report.errors.append(ValidationIssue(
                severity="error",
                component="storage",
                field="directory",
                message="Storage configuration directory not found",
                file_path=str(storage_dir)
            ))
            return
        
        # Validate storage config files
        for storage_file in storage_dir.glob("*.yaml"):
            self._validate_storage_file(storage_file)
    
    def _validate_flow_configs(self):
        """Validate flow-specific configurations."""
        flows_dir = self.base_path / "flows"
        if not flows_dir.exists():
            self.report.errors.append(ValidationIssue(
                severity="error",
                component="flows",
                field="directory", 
                message="Flows configuration directory not found",
                file_path=str(flows_dir)
            ))
            return
        
        for flow_file in flows_dir.glob("*.yaml"):
            self._validate_flow_file(flow_file)
    
    def _validate_yaml_file(self, file_path: Path, component: str):
        """Validate a YAML configuration file."""
        if not file_path.exists():
            self.report.errors.append(ValidationIssue(
                severity="error",
                component=component,
                field="file",
                message=f"Configuration file not found: {file_path.name}",
                file_path=str(file_path)
            ))
            return
        
        try:
            import yaml
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                self.report.warnings.append(ValidationIssue(
                    severity="warning",
                    component=component,
                    field="content",
                    message=f"Configuration file is empty: {file_path.name}",
                    file_path=str(file_path)
                ))
            else:
                self.report.passed_checks.append(f"Valid YAML: {file_path.name}")
                
        except yaml.YAMLError as e:
            self.report.errors.append(ValidationIssue(
                severity="error",
                component=component,
                field="syntax",
                message=f"YAML syntax error in {file_path.name}: {str(e)}",
                file_path=str(file_path),
                suggestion="Check YAML syntax and indentation"
            ))
        except Exception as e:
            self.report.errors.append(ValidationIssue(
                severity="error",
                component=component,
                field="read",
                message=f"Failed to read {file_path.name}: {str(e)}",
                file_path=str(file_path)
            ))
        
        self.report.total_files_checked += 1
    
    def _validate_storage_file(self, file_path: Path):
        """Validate a storage configuration file."""
        try:
            import yaml
            with open(file_path, 'r') as f:
                storage_config = yaml.safe_load(f)
            
            if not storage_config:
                return
            
            # Check for required storage fields
            if 'type' not in storage_config:
                self.report.errors.append(ValidationIssue(
                    severity="error",
                    component="storage",
                    field="type",
                    message=f"Storage type not specified in {file_path.name}",
                    file_path=str(file_path),
                    suggestion="Add 'type' field with value like 'bigquery', 'file', 'chromadb', etc."
                ))
            else:
                storage_type = storage_config['type']
                self._validate_storage_type_config(storage_type, storage_config, file_path)
            
            self.report.passed_checks.append(f"Storage config structure valid: {file_path.name}")
            
        except Exception as e:
            self.report.errors.append(ValidationIssue(
                severity="error",
                component="storage",
                field="validation",
                message=f"Failed to validate storage config {file_path.name}: {str(e)}",
                file_path=str(file_path)
            ))
        
        self.report.total_files_checked += 1
    
    def _validate_storage_type_config(self, storage_type: str, config: Dict[str, Any], file_path: Path):
        """Validate type-specific storage configuration."""
        type_requirements = {
            "bigquery": ["project_id", "dataset_id", "table_id"],
            "file": ["path"],
            "chromadb": ["collection_name", "persist_directory"],
            "vector": ["collection_name"],
            "huggingface": ["dataset_id"],
        }
        
        if storage_type in type_requirements:
            required_fields = type_requirements[storage_type]
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                self.report.warnings.append(ValidationIssue(
                    severity="warning",
                    component="storage",
                    field=", ".join(missing_fields),
                    message=f"Missing recommended fields for {storage_type} storage: {missing_fields}",
                    file_path=str(file_path),
                    suggestion=f"Consider adding: {', '.join(missing_fields)}"
                ))
        
        # Check for common issues
        if storage_type == "chromadb" and "dimensionality" in config:
            self.report.info.append(ValidationIssue(
                severity="info",
                component="storage",
                field="dimensionality",
                message="ChromaDB storage config has dimensionality field",
                file_path=str(file_path),
                suggestion="Dimensionality is automatically detected from embedding model"
            ))
    
    def _validate_flow_file(self, file_path: Path):
        """Validate a flow configuration file."""
        try:
            import yaml
            with open(file_path, 'r') as f:
                flow_config = yaml.safe_load(f)
            
            if not flow_config:
                return
                
            # Validate agents configuration
            if 'agents' in flow_config:
                agents = flow_config['agents']
                for agent_name, agent_config in agents.items():
                    self._validate_agent_config(agent_name, agent_config, file_path)
            
            # Validate data sources
            if 'data' in flow_config:
                data_sources = flow_config['data']
                for data_name, data_config in data_sources.items():
                    self._validate_data_source_config(data_name, data_config, file_path)
            
            self.report.passed_checks.append(f"Flow config structure valid: {file_path.name}")
            
        except Exception as e:
            self.report.errors.append(ValidationIssue(
                severity="error",
                component="flows",
                field="validation",
                message=f"Failed to validate flow config {file_path.name}: {str(e)}",
                file_path=str(file_path)
            ))
        
        self.report.total_files_checked += 1
    
    def _validate_agent_config(self, agent_name: str, agent_config: Dict[str, Any], file_path: Path):
        """Validate an agent configuration."""
        required_fields = ["role", "agent_obj"]
        missing_fields = [field for field in required_fields if field not in agent_config]
        
        if missing_fields:
            self.report.errors.append(ValidationIssue(
                severity="error",
                component="agent",
                field=f"{agent_name}.{missing_fields[0]}",
                message=f"Agent {agent_name} missing required fields: {missing_fields}",
                file_path=str(file_path),
                suggestion=f"Add required fields: {', '.join(missing_fields)}"
            ))
        
        # Check for Enhanced RAG agent specific configuration
        if agent_config.get("agent_obj") == "EnhancedRagAgent":
            self._validate_enhanced_rag_config(agent_name, agent_config, file_path)
    
    def _validate_enhanced_rag_config(self, agent_name: str, agent_config: Dict[str, Any], file_path: Path):
        """Validate Enhanced RAG agent specific configuration."""
        # Check for data source
        if "data" not in agent_config:
            self.report.warnings.append(ValidationIssue(
                severity="warning",
                component="enhanced_rag",
                field=f"{agent_name}.data",
                message=f"Enhanced RAG agent {agent_name} has no data sources configured",
                file_path=str(file_path),
                suggestion="Enhanced RAG agents need vector storage data sources"
            ))
        
        # Check parameters
        params = agent_config.get("parameters", {})
        recommended_params = ["enable_query_planning", "enable_result_synthesis", "search_strategies"]
        
        for param in recommended_params:
            if param not in params:
                self.report.info.append(ValidationIssue(
                    severity="info",
                    component="enhanced_rag",
                    field=f"{agent_name}.parameters.{param}",
                    message=f"Enhanced RAG agent {agent_name} could benefit from {param} configuration",
                    file_path=str(file_path),
                    suggestion=f"Consider adding {param} to parameters"
                ))
    
    def _validate_data_source_config(self, data_name: str, data_config: Dict[str, Any], file_path: Path):
        """Validate a data source configuration."""
        if "_target_" not in data_config:
            self.report.warnings.append(ValidationIssue(
                severity="warning",
                component="data_source",
                field=f"{data_name}._target_",
                message=f"Data source {data_name} missing _target_ field",
                file_path=str(file_path),
                suggestion="Add _target_ field pointing to storage config"
            ))
    
    def _generate_summary(self):
        """Generate validation summary."""
        logger.info(f"Configuration validation complete:")
        logger.info(f"  Files checked: {self.report.total_files_checked}")
        logger.info(f"  Errors: {len(self.report.errors)}")
        logger.info(f"  Warnings: {len(self.report.warnings)}")
        logger.info(f"  Info issues: {len(self.report.info)}")
        logger.info(f"  Passed checks: {len(self.report.passed_checks)}")
        logger.info(f"  Dependency issues: {len(self.report.dependency_issues)}")
    
    def validate_pydantic_models(self) -> List[ValidationIssue]:
        """Test validation of core Pydantic models with sample data."""
        issues = []
        
        try:
            from buttermilk._core.storage_config import (
                VectorStorageConfig, 
                FileStorageConfig, 
                BigQueryStorageConfig,
                StorageFactory
            )
            
            # Test type-specific storage configs
            test_configs = [
                {"type": "chromadb", "collection_name": "test"},
                {"type": "file", "path": "/test/path"},
                {"type": "bigquery", "project_id": "test", "dataset_id": "test", "table_id": "test"}
            ]
            
            for config_data in test_configs:
                try:
                    config = StorageFactory.create_config(config_data)
                    logger.debug(f"Successfully validated {config_data['type']} storage config")
                except ValidationError as e:
                    issues.append(ValidationIssue(
                        severity="error",
                        component="pydantic_model",
                        field=f"{config_data['type']}_storage_config",
                        message=f"Pydantic validation failed for {config_data['type']}: {str(e)}",
                        suggestion="Check field types and required values"
                    ))
                except Exception as e:
                    issues.append(ValidationIssue(
                        severity="warning",
                        component="pydantic_model", 
                        field=f"{config_data['type']}_storage_config",
                        message=f"Unexpected error validating {config_data['type']}: {str(e)}"
                    ))
            
        except ImportError as e:
            issues.append(ValidationIssue(
                severity="warning",
                component="pydantic_model",
                field="imports",
                message=f"Could not import storage config models: {str(e)}",
                suggestion="Check that all dependencies are available"
            ))
        
        return issues


def validate_configuration(config_path: str = "/workspaces/buttermilk/conf") -> ValidationReport:
    """
    Run comprehensive configuration validation.
    
    Args:
        config_path: Path to configuration directory
        
    Returns:
        Validation report with all issues found
    """
    validator = ConfigValidator(config_path)
    report = validator.validate_all()
    
    # Add Pydantic model validation
    model_issues = validator.validate_pydantic_models()
    for issue in model_issues:
        if issue.severity == "error":
            report.errors.append(issue)
        elif issue.severity == "warning":
            report.warnings.append(issue)
        else:
            report.info.append(issue)
    
    return report


if __name__ == "__main__":
    # Quick validation check
    print("🔧 CONFIGURATION VALIDATION")
    print("=" * 40)
    
    report = validate_configuration()
    
    print(f"Files checked: {report.total_files_checked}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")
    print(f"Dependency issues: {len(report.dependency_issues)}")
    
    if report.errors:
        print("\n❌ ERRORS:")
        for error in report.errors[:5]:
            print(f"  {error.component}.{error.field}: {error.message}")
    
    if report.warnings:
        print("\n⚠️ WARNINGS:")
        for warning in report.warnings[:5]:
            print(f"  {warning.component}.{warning.field}: {warning.message}")
    
    if report.is_valid:
        print("\n✅ Configuration validation passed!")
    else:
        print(f"\n❌ Configuration validation failed with {len(report.errors)} errors")