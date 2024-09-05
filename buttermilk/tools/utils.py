from pathlib import Path

from ruamel.yaml import YAML


def collect_tools_from_directory(base_dir) -> dict:
    tools = {}
    yaml = YAML()
    for f in Path(base_dir).glob("**/*.yaml"):
        with open(f, "r") as f:
            tools_in_file = yaml.load(f)
            for identifier, tool in tools_in_file.items():
                tools[identifier] = tool
    return tools


def list_package_tools():
    """List package tools"""
    yaml_dir = Path(__file__).parent / "yamls"
    return collect_tools_from_directory(yaml_dir)

if __name__ == '__main__':
    print(list_package_tools())