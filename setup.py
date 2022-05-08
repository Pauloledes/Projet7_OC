import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="P7_pledes_api",
    version="0.1",
    author="Paul ledesma",
    author_email="paul.ledesma@hotmail.fr",
    description="Description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pauloledes/Projet7_OC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)