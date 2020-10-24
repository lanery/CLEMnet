from setuptools import setup, find_packages

DISTNAME = 'clemnet'
DESCRIPTION = 'secdetect: Convolutional neural network for predicting fluorescence from EM images'
MAINTAINER = 'Ryan Lane'
MAINTAINER_EMAIL = 'r.i.lane@tudelft.nl'
LICENSE = 'LICENSE'
README = 'README.md'
URL = 'https://github.com/lanery/clemnet'
VERSION = '0.1.dev'
PACKAGES = [
    'clemnet',
]
INSTALL_REQUIRES = [
    # 'tensorflow',
]

if __name__ == '__main__':

    setup(
        name=DISTNAME,
        version=VERSION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        packages=PACKAGES,
        include_package_data=True,
        url=URL,
        license=LICENSE,
        description=DESCRIPTION,
        long_description=open(README).read(),
        install_requires=INSTALL_REQUIRES,
    )
