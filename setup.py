from distutils.core import setup
from setuptools import find_packages
from resource_manager import __version__

classifiers = '''Development Status :: 5 - Production/Stable
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9'''

with open('README.rst', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='resource-manager',
    packages=find_packages(include=('resource_manager',)),
    include_package_data=True,
    version=__version__,
    description='A small resource manager for config files',
    long_description=long_description,
    author='maxme1',
    author_email='maxs987@gmail.com',
    license='MIT',
    url='https://github.com/maxme1/resource-manager',
    download_url='https://github.com/maxme1/resource-manager/archive/%s.tar.gz' % __version__,
    keywords=[
        'config', 'lazy', 'interpreter'
    ],
    classifiers=classifiers.splitlines(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'run-config = resource_manager.console:render_config_resource',
            'build-config = resource_manager.console:build_config'
        ],
    },
)
