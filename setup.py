import os

import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))


# Load README.
def readme():
    readme_path = os.path.join(this_directory, "README.md")
    with open(readme_path, encoding="utf-8") as fp:
        return fp.read()


# Load requirements.
def requirements():
    requirements_path = os.path.join(this_directory, "requirements.txt")
    with open(requirements_path, encoding="utf-8") as fp:
        return fp.read().splitlines()


setuptools.setup(
    name="diffusion",
    version="0.0.1",
    author="Maruan Al-Shedivat",
    author_email="maruan@alshedivat.com",
    description="Playground for experimenting with diffusion generative models.",
    long_description=readme(),
    url="https://github.com/alshedivat/diffusion-playground",
    install_requires=requirements(),
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
