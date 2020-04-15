import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SGHMC", # Replace with your own username
    version="0.0.2",
    author="Zining Ma, Machao Deng",
    author_email="zining.ma@duke.edu, machao.deng@duke.edu",
    description="Implementation of Stochastic Gradient Hamiltonian Monte Carlo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Senlody/SGHMC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License ::  OSI Approved:: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy >= 1.14.2",
        "seaborn >= 0.8.1",
        "matplotlib >= 2.0.0",
        "scipy >= 1.0.1",
        "numba >= 0.37.0"
    ],
)