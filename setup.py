import setuptools


def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setuptools.setup(
    name="wavencoder",
    author="Shangeth Rajaa",
    author_email="shangethrajaa@gmail.com",
    version="0.0.1",
    license="MIT",
    url="https://github.com/shangeth/wavencoder",
    description="WavEncoder - PyTorch backed audio encoder",
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires='>=3.5',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=["torch", "tqdm"],
# line.strip() for line in open("requirements.txt", "r").readlines()],
    # dependency_links=["git+https://github.com/pytorch/fairseq"]
)