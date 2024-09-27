# from gettext import install
# from pkg_resources import re
from setuptools import find_packages, setup
from typing import List

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

setup( 
    name='Evaluate_XAI_ArabicNLP',
    version='0.0.1',
    author='Yousra',
    author_email='hadjazzem.yousra@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )