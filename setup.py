from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="mra",
    version="0.0.1",
    packages=["mra"],
    license="GPLv3",
    description="Multiple-response agents: Fast, feasible, approximate primal recovery for dual optimization methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # install_requires=[
    #     "numpy >= 1.22.2",
    #     "scipy >= 1.8.0",
    #     "cvxpy >= 1.2.0",
    #     "matplotlib >= 3.5.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)