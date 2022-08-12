from setuptools import find_packages, setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='AudioStudio',
    packages=find_packages(include=['AudioStudio']),
    version='0.0.4',
    description='A library for working with audio data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Daniel Rabayda',
    author_email='rabaydadp@gmail.com',
    license='MIT',
    url='https://github.com/danrabayda/AudioStudio',
    install_requires=[
        'pydub',
        'ipywidgets',
        'numpy',
        'scipy',
        'matplotlib',
        'IPython',
      ],
)
