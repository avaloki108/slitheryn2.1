from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="slitheryn-analyzer",
    description="Slitheryn is a Solidity and Vyper static analysis framework written in Python 3.",
    url="https://github.com/avaloki108/slitheryn2.1",
    author="Trail of Bits",
    version="0.11.3",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "packaging",
        "prettytable>=3.10.2",
        "pycryptodome>=3.4.6",
        "crytic-compile>=0.3.9,<0.4.0",
        # "crytic-compile@git+https://github.com/crytic/crytic-compile.git@master#egg=crytic-compile",
        "web3>=7.10,<8",
        "eth-abi>=5.0.1",
        "eth-typing>=5.0.0",
        "eth-utils>=5.0.0",
    ],
    extras_require={
        "lint": [
            "black==22.3.0",
            "pylint==3.0.3",
        ],
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "deepdiff",
            "orderly-set==5.3.2",  # Temporary fix for https://github.com/seperman/deepdiff/issues/539
            "numpy",
            "coverage[toml]",
            "filelock",
            "pytest-insta",
        ],
        "doc": [
            "pdoc",
        ],
        "dev": [
            "slitheryn-analyzer[lint,test,doc]",
            "openai",
        ],
    },
    license="AGPL-3.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "slitheryn = slitheryn.__main__:main",
            "slitheryn-cli = slitheryn_cli:main",
            "slitheryn-check-upgradeability = slitheryn.tools.upgradeability.__main__:main",
            "slitheryn-find-paths = slitheryn.tools.possible_paths.__main__:main",
            "slitheryn-simil = slitheryn.tools.similarity.__main__:main",
            "slitheryn-flat = slitheryn.tools.flattening.__main__:main",
            "slitheryn-format = slitheryn.tools.slitheryn_format.__main__:main",
            "slitheryn-check-erc = slitheryn.tools.erc_conformance.__main__:main",
            "slitheryn-check-kspec = slitheryn.tools.kspec_coverage.__main__:main",
            "slitheryn-prop = slitheryn.tools.properties.__main__:main",
            "slitheryn-mutate = slitheryn.tools.mutator.__main__:main",
            "slitheryn-read-storage = slitheryn.tools.read_storage.__main__:main",
            "slitheryn-doctor = slitheryn.tools.doctor.__main__:main",
            "slitheryn-documentation = slitheryn.tools.documentation.__main__:main",
            "slitheryn-interface = slitheryn.tools.interface.__main__:main",
        ]
    },
)
