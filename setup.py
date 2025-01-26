from setuptools import setup, find_packages

# Read the content of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="19F_NMR_spectrum_predictor_explore",
    version="0.1",  # Project version
    author="Dandan Rao",
    author_email="kiluarao@gmail.com",
    description="A project developed to predict 19F NMR spectra for fluorinated compounds, especially PFAS, using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically finds all packages in the directory

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

