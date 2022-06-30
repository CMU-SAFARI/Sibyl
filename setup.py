import os
import re
from subprocess import check_call
from setuptools import setup, find_packages
from setuptools.command.install import install


__pkg_name__ = 'sibyl'

verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))


require_file = 'requirements.txt'
package_name = "safari-%s" % __pkg_name__

with open(require_file) as f:
    requirements = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='CMU-SAFARI',
    author_email='gagsingh@ethz.ch',
    url='https://github.com/CMU-SAFARI/Sibyl',
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    }
)