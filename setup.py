
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()    

setup(
    name="movingfp",
    version="0.2.1",
    license='MIT',
    author="Network and Data Science Laboratory CIC-IPN México",
    author_email="omar.gup@gmail.com",
    description="Moving Fire Fighter Problem Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omargup/firefighter_problem_generator",
    keywords='moving firefighter problem',
    packages=["movingfp"],
    python_requires='>=3.8',
    install_requires=['numpy', 'networkx'])