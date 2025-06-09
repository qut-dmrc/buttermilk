"""Example setup.py for external research configuration package."""

from setuptools import setup, find_packages

setup(
    name="example_research_configs",
    version="0.1.0",
    description="Example Buttermilk configurations for research project",
    packages=find_packages(),
    package_data={
        "example_research_configs": ["conf/**/*.yaml", "templates/**/*"],
    },
    include_package_data=True,
    install_requires=[
        "buttermilk",  # Require the main buttermilk package
    ],
    python_requires=">=3.10",
    author="Example Researcher",
    author_email="researcher@university.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)