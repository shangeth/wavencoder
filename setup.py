import setuptools

setuptools.setup(
    name='wavencoder',
    version='0.0.8',
    description='WavEncoder - PyTorch backed audio encoder',
    packages=setuptools.find_packages(),
    install_requires=['torch', 'wget', 'fairseq']
)