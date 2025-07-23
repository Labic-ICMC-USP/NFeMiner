from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'NFeMiner'
LONG_DESCRIPTION = 'NFeMiner'

setup(
        name="nfeminer", 
        version=VERSION,
        author="Labic",
        author_email="<youremail@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['NFe']
)
