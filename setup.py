from setuptools import find_packages, setup

with open("./requirements.txt", "r") as f:
    requirements = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

setup(
    name="hm3d-data-collector",
    version="2.1.3",
    packages=find_packages(),
    install_requires=requirements,
    package_data={
        "hm3d_data_collector": ["config/**"]
    },
    author="Enrique Solarte",
    author_email="enrique.solarte.pardo@gmail.com",
    description=("This is data collector for HM3D dataset"),
    license="BSD",
)
