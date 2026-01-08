from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->  List[str]:
    """
    Docstring for get_requirements that will retirn a list of requirments
        
    :param file_path: Description
    :type file_path: str
    :return: Description
    :rtype: List[str]
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="Shield Insurance premium prediction",
    version="0.0.1",
    author="Erick Yegon",
    author_email="keyegonaws@gmail.com",
    description="A production-ready, end-to-end machine learning system for predicting health insurance premiums using demographic, lifestyle, and medical risk factors, featuring domain-driven feature engineering, multicollinearity control, and cloud-ready deployment.",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)