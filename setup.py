
from setuptools import find_packages, setup

setup(
    name="walkingffp",
    version="0.0.1",
    license='MIT',
    author="Omar Gutiérrez",
    author_email="omar.gup@gmail.com",
    description="Moving/Walking Fire Fighter Problem Generator",
    long_description="Instances generator of the Moving/Walking Fire Fighter Problem, proposed by the Network and Data Science Laboratory CIC-IPN México.",
    long_description_content_type="text/markdown",
    url="https://github.com/omargup/firefighter_problem_generator",
    keywords='walking firefighter problem',
    packages=["walkingffp"],
    python_requires='>=3.8',
    install_requires=['numpy', 'networkx'])