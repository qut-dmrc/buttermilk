#!/bin/bash

# Buttermilk Configuration Validator Tool
# Validate YAML configurations and Hydra interpolations

CONFIG_PATH="$1"
CHECK_INTERPOLATIONS="${2:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 Buttermilk Config Validator${NC}"
echo "=============================="
echo ""

if [ -z "$CONFIG_PATH" ]; then
    echo -e "${RED}❌ Missing config path${NC}"
    echo "Usage: buttermilk-config-validate <config_path> [check_interpolations]"
    echo "  config_path: Path to config file relative to /src/buttermilk"
    echo "  check_interpolations: true/false (default: true)"
    exit 1
fi

# Construct full path
FULL_PATH="/src/buttermilk/$CONFIG_PATH"

if [ ! -f "$FULL_PATH" ]; then
    echo -e "${RED}❌ Config file not found: $FULL_PATH${NC}"
    echo ""
    echo "Available config files:"
    find /src/buttermilk/conf -name "*.yaml" -o -name "*.yml" | head -10
    exit 1
fi

echo -e "Config file: ${BLUE}$FULL_PATH${NC}"
echo -e "Check interpolations: ${BLUE}$CHECK_INTERPOLATIONS${NC}"
echo ""

# Create Python validation script
cat > /tmp/validate_config.py << 'EOF'
#!/usr/bin/env python3
import sys
import yaml
from pathlib import Path
import re
from typing import Dict, List, Any, Set

class ConfigValidator:
    def __init__(self, config_path: str, check_interpolations: bool = True):
        self.config_path = Path(config_path)
        self.check_interpolations = check_interpolations
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.interpolations: Set[str] = set()
        
    def validate(self) -> bool:
        """Validate the configuration file."""
        try:
            # Load YAML
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                self.errors.append("Empty configuration file")
                return False
                
            # Validate structure
            self._validate_structure(config)
            
            # Check interpolations if requested
            if self.check_interpolations:
                self._check_interpolations(config)
                
            # Check for common issues
            self._check_common_issues(config)
            
            return len(self.errors) == 0
            
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")
            return False
            
    def _validate_structure(self, config: Dict[str, Any], path: str = "") -> None:
        """Validate the configuration structure."""
        if not isinstance(config, dict):
            return
            
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            # Check for required fields based on config type
            if key == "agents" and isinstance(value, list):
                for i, agent in enumerate(value):
                    agent_path = f"{current_path}[{i}]"
                    self._validate_agent(agent, agent_path)
                    
            elif key == "flows" and isinstance(value, dict):
                for flow_name, flow_config in value.items():
                    flow_path = f"{current_path}.{flow_name}"
                    self._validate_flow(flow_config, flow_path)
                    
            # Recurse for nested structures
            if isinstance(value, dict):
                self._validate_structure(value, current_path)
                
    def _validate_agent(self, agent: Dict[str, Any], path: str) -> None:
        """Validate agent configuration."""
        required_fields = ["_target_", "agent_id"]
        
        for field in required_fields:
            if field not in agent:
                self.errors.append(f"Missing required field '{field}' at {path}")
                
        # Check _target_ is valid
        if "_target_" in agent:
            target = agent["_target_"]
            if not target.startswith("buttermilk."):
                self.warnings.append(f"Unusual agent target at {path}: {target}")
                
    def _validate_flow(self, flow: Dict[str, Any], path: str) -> None:
        """Validate flow configuration."""
        if "participants" not in flow:
            self.errors.append(f"Missing 'participants' in flow at {path}")
            
    def _check_interpolations(self, config: Any, path: str = "") -> None:
        """Check Hydra interpolations."""
        if isinstance(config, str):
            # Find interpolations
            interpolations = re.findall(r'\$\{([^}]+)\}', config)
            for interp in interpolations:
                self.interpolations.add(interp)
                
                # Check for common issues
                if ".." in interp:
                    self.warnings.append(f"Double dots in interpolation at {path}: ${{{interp}}}")
                    
        elif isinstance(config, dict):
            for key, value in config.items():
                current_path = f"{path}.{key}" if path else key
                self._check_interpolations(value, current_path)
                
        elif isinstance(config, list):
            for i, item in enumerate(config):
                self._check_interpolations(item, f"{path}[{i}]")
                
    def _check_common_issues(self, config: Dict[str, Any]) -> None:
        """Check for common configuration issues."""
        # Check for hardcoded values that should be interpolated
        hardcoded_patterns = [
            (r'(localhost|127\.0\.0\.1|0\.0\.0\.0):\d+', "Hardcoded host/port"),
            (r'(sk-[a-zA-Z0-9]+|key-[a-zA-Z0-9]+)', "Potential API key"),
            (r'(/home/[^/]+|/Users/[^/]+)', "Hardcoded user path"),
        ]
        
        config_str = str(config)
        for pattern, issue in hardcoded_patterns:
            if re.search(pattern, config_str):
                self.warnings.append(f"{issue} detected - consider using interpolation")
                
    def print_report(self) -> None:
        """Print validation report."""
        if self.errors:
            print(f"\033[0;31m❌ Validation failed with {len(self.errors)} error(s)\033[0m")
            print("\nErrors:")
            for error in self.errors:
                print(f"  • {error}")
        else:
            print("\033[0;32m✅ Configuration is valid\033[0m")
            
        if self.warnings:
            print(f"\n\033[1;33m⚠️  {len(self.warnings)} warning(s)\033[0m")
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        if self.interpolations:
            print(f"\n\033[0;34m🔗 Found {len(self.interpolations)} interpolation(s)\033[0m")
            print("\nInterpolations:")
            for interp in sorted(self.interpolations):
                print(f"  • ${{{interp}}}")

if __name__ == "__main__":
    config_path = sys.argv[1]
    check_interpolations = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else True
    
    validator = ConfigValidator(config_path, check_interpolations)
    is_valid = validator.validate()
    validator.print_report()
    
    sys.exit(0 if is_valid else 1)
EOF

# Run the validation
echo -e "${BLUE}Running validation...${NC}"
echo ""

if uv run python /tmp/validate_config.py "$FULL_PATH" "$CHECK_INTERPOLATIONS"; then
    EXIT_CODE=0
else
    EXIT_CODE=1
fi

# Additional checks using config_validator if available
if [ -f "/src/buttermilk/buttermilk/utils/config_validator.py" ]; then
    echo ""
    echo -e "${BLUE}Running Buttermilk config validator...${NC}"
    echo ""
    
    cd /src/buttermilk
    if uv run python -m buttermilk.utils.config_validator "$CONFIG_PATH" 2>&1; then
        echo -e "${GREEN}✅ Buttermilk validator passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Buttermilk validator reported issues${NC}"
        EXIT_CODE=1
    fi
fi

# Cleanup
rm -f /tmp/validate_config.py

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Configuration validation completed successfully${NC}"
else
    echo -e "${RED}❌ Configuration validation failed${NC}"
fi

exit $EXIT_CODE