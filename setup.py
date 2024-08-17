#creating the ML Aplication as a pacakage
from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT="-e ."

def get_requirements(file_path:str)->List[str]:
    """
    this function will return a list of requirements
    """

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
name="mlproject",
version="0.0.1",
author="Jivittesh",
author_email="jivittesh@gmail.com",
packages=find_packages(),
install_requires=get_requirements(r'requirements.txt')
)



#src to be fund as a package --> when find pacakages is running,it
# will find how many files have this __init__.py and src will be considered as  apackage and build src and install pacakges

##requirements.txt   --> -e. to connect to setup.py