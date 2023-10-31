"""
Setup file, used to create package of this entire folder

Author: Haider Jan
Creation Date: 30/10/2023
"""
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    this function will return the list of requirements
    """
    requirements = []
    with open("requirements.txt") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="udacity_assignment",
    version="0.0.1",
    author="Haider",
    author_email="janhaider040@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements_py3.8.txt"),
)
