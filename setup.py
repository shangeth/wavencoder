import setuptools

setuptools.setup(
    name='wavencoder',
    version='0.1.3',
    description='WavEncoder - PyTorch backed audio encoder',
    packages=setuptools.find_packages(),
    install_requires=['torch', 'fairseq'],
    dependency_links=[]
)