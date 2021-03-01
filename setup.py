import setuptools


def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setuptools.setup(
    name="wavencoder",
    author="Shangeth Rajaa",
    author_email="shangethrajaa@gmail.com",
    version="0.1.2", 
    license="MIT",
    url="https://github.com/shangeth/wavencoder",
    description="WavEncoder - PyTorch backed audio encoder",
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[ 
        "numpy>=1.19.2",
        "torch>=1.6.0",
        "torchaudio>=0.6.0", 
        "tqdm",
        "fairseq"
],
)
