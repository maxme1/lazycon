import runpy
from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3 :: Only',
]

name = 'lazycon'
root = Path(__file__).resolve().parent
with open(root / 'README.md', encoding='utf-8') as file:
    long_description = file.read()
version = runpy.run_path(root / name / '__version__.py')['__version__']

setup(
    name=name,
    packages=find_packages(include=(name,)),
    include_package_data=True,
    version=version,
    description='Easy config files in pure Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Max',
    author_email='max@ira-labs.com',
    license='MIT',
    url='https://github.com/maxme1/lazycon',
    download_url='https://github.com/maxme1/lazycon/archive/v%s.tar.gz' % version,
    keywords=['config', 'lazy', 'interpreter'],
    classifiers=classifiers,
    install_requires=[],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'lazycon = lazycon.console:main',
        ],
    },
)
