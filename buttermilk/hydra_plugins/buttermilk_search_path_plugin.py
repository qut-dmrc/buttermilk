"""Hydra SearchPath plugin for automatic discovery of Buttermilk configurations.

This plugin automatically adds common configuration search paths to Hydra's
search path, enabling external users to define their own configurations
without modifying the buttermilk source code.
"""

import os
from pathlib import Path
from typing import List

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class ButtermilkSearchPathPlugin(SearchPathPlugin):
    """Plugin to automatically discover Buttermilk configuration directories.
    
    This plugin adds the following search paths in order:
    1. Core buttermilk configurations (highest priority for base configs)
    2. User-specific configurations from environment variables
    3. Project-specific configurations from well-known locations
    4. External package configurations (e.g., installed research packages)
    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Add configuration search paths for Buttermilk.
        
        Args:
            search_path: Hydra's ConfigSearchPath to modify
        """
        # 1. Core buttermilk configurations (base infrastructure)
        search_path.append("buttermilk_core", "pkg://buttermilk.conf")

        # 2. User-specific configuration directory
        user_config_dir = os.getenv("BUTTERMILK_CONFIG_DIR")
        if user_config_dir and Path(user_config_dir).exists():
            search_path.append("user_config", f"file://{user_config_dir}")

        # 3. Project-specific configurations in current working directory
        cwd_conf = Path.cwd() / "conf"
        if cwd_conf.exists():
            search_path.append("project_config", f"file://{cwd_conf}")

        # 4. Home directory configurations
        home_config = Path.home() / ".config" / "buttermilk"
        if home_config.exists():
            search_path.append("home_config", f"file://{home_config}")

        # 5. Well-known external package patterns
        external_packages = self._discover_external_packages()
        for pkg_name, pkg_path in external_packages:
            search_path.append(f"external_{pkg_name}", pkg_path)

    def _discover_external_packages(self) -> List[tuple[str, str]]:
        """Discover external packages that provide Buttermilk configurations.
        
        Looks for packages with naming patterns like:
        - buttermilk_*_configs
        - *_buttermilk_flows
        - research_*_configs
        
        Returns:
            List of (package_name, package_path) tuples
        """
        external_packages = []

        # Common patterns for research configuration packages
        patterns = [
            "buttermilk_*_configs",
            "*_buttermilk_flows",
            "research_*_configs",
            "*_research_flows"
        ]

        try:
            import importlib.util
            import sys

            # Check for installed packages matching our patterns
            for module_name in sys.modules:
                if any(self._matches_pattern(module_name, pattern) for pattern in patterns):
                    try:
                        spec = importlib.util.find_spec(f"{module_name}.conf")
                        if spec is not None:
                            external_packages.append((module_name, f"pkg://{module_name}.conf"))
                    except ImportError:
                        continue

        except Exception:
            # If package discovery fails, continue without external packages
            pass

        return external_packages

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a module name matches a wildcard pattern."""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)
