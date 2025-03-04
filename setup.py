# setup.py is the fallback installation script when pyproject.toml does not work
from setuptools import setup, find_packages
import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# with open(os.path.join(version_folder, 'version/version')) as f:
#     __version__ = f.read().strip()


with open('requirements.txt') as f:
    required = f.read().splitlines()
    install_requires = [item.strip() for item in required if item.strip()[0] != '#']

extras_require = {
    'test': ['yapf']
}

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='PhysRes',
    #version=__version__,
    version='0.0.1',
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    #url='https://github.com/',
    #license='Apache 2.0',
    author='Cedric Caremel',
    author_email='cedric@keio.jp',
    description='Physical Reservoir Computing',
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={'': ['version/*']},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)