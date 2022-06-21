from distutils.core import setup
from setuptools import find_packages
from lazycon import __version__

classifiers = '''Development Status :: 5 - Production/Stable
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9'''

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='lazycon',
    packages=find_packages(include=('lazycon',)),
    include_package_data=True,
    version=__version__,
    description='Easy config files in pure Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='maxme1',
    author_email='maxs987@gmail.com',
    license='MIT',
    url='https://github.com/maxme1/lazycon',
    download_url='https://github.com/maxme1/lazycon/archive/v%s.tar.gz' % __version__,
    keywords=[
        'config', 'lazy', 'interpreter'
    ],
    classifiers=classifiers.splitlines(),
    install_requires=[],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'lazycon = lazycon.console:main',
        ],
    },
)
