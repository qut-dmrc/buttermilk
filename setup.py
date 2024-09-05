from setuptools import find_packages, setup

PACKAGE_NAME = "buttermilk"

setup(
    name=PACKAGE_NAME,
    version="0.0.13",
    description="This is my tools package",
    packages=find_packages(),
    entry_points={
    },
    include_package_data=True,   # This line tells setuptools to include files from MANIFEST.in
    extras_require={
        "azure": [
            "azure-ai-ml>=1.11.0,<2.0.0"
        ]
    },
)