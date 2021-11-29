import os

import setuptools


def select_all_folders(root):
    if root is None:
        raise TypeError("root folder can't be None")

    return [x[0] for x in os.walk(root)]


def setup_version() -> str:
    version_prefix = "0.1"
    version = os.environ.get("BUILD_NUMBER") if bool(os.environ.get("TEAMCITY_VERSION")) else "SNAPSHOT"
    return f"{version_prefix}.{version}"


if os.path.exists("README.md"):
    with open("README.md", "r") as fh:
        long_description = fh.read()
else:
    long_description = ""

setuptools.setup(
    name="flccpsi",
    version=setup_version(),
    author="Yaroslav Sokolov",
    author_email="sokolov.yas@gmail.com",
    description="Full Line Code generation via beam search of PSI prediction models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SokolovYaroslav/PSI-Transformer",
    packages=select_all_folders("flccpsi") + select_all_folders("flccpsisrc"),
    nclude_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
)
