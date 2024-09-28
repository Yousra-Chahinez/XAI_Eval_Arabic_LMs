# from gettext import install
# from pkg_resources import re
from setuptools import find_packages, setup
from typing import List
import pkg_resources


HYPEN_E_DOT="-e ."

def get_requirements(file_path:str)->List[str]:
    """
    return a list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT) # -e . trigger setup.py
    return requirements


# Read your existing requirements file
with open('requirements.txt', 'r') as f:
    required_packages = [line.strip().split('==')[0] for line in f.readlines() if not line.startswith('-e')]

# Get the versions of installed packages
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# Write the requirements back with versions
with open('requirements.txt', 'w') as f:
    for package in required_packages:
        if package in installed_packages:
            f.write(f"{package}=={installed_packages[package]}\n")
        else:
            f.write(f"{package}\n")  # Keep packages that don't have versions
    
    # Keep editable packages at the end if any
    f.write('-e .\n')


setup( 
    name='Evaluate_XAI_ArabicNLP',
    version='0.0.1',
    author='Yousra',
    author_email='hadjazzem.yousra@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )