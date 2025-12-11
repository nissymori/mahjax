from setuptools import find_packages, setup
from pathlib import Path

setup(
    name="mahjax",
    description="GPU-accelerated vectorized mahjong simulators for reinforcement learning",
    long_description_content_type="text/markdown",
    author="Soichiro Nishimori",
    author_email="gatikiti.630@gmail.com",
    keywords="",
    packages=find_packages(),
    package_data={
        "": ["LICENSE", "*.svg"]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)