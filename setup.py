'''
This file is used to package the project and install the dependencies.  
'''

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    Read the requirements.txt file and return the dependencies as a list of strings.
    """
    requirements: List[str] = []
    try:
        with open('requirements.txt', 'r') as f:
            # Read and filter lines, ignoring empty lines and '-e' entries
            for line in f:
                requirement = line.strip()
                if requirement and not requirement.startswith('-e'):
                    requirements.append(requirement)
    except FileNotFoundError:
        print("Requirements file not found.")

    return requirements

setup(
    name="Telco Customer Churn Prediction",
    version="0.0.1",
    author="Daniyal Mufti",
    author_email="seether124@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()    
)                    