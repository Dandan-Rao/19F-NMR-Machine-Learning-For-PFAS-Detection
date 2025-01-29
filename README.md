# Recommended Running Environment - GitHub Codespaces
* 1. Fork the repository.

* 2. Open GitHub Codespaces by clicking:

'Code' → 'Codespaces' → 'Create Codespace on main'

* 3. Once inside the Codespace, create a virtual environment by running the following commands in the terminal:

```
$ python -m venv .venv
$ source .venv/bin/activate
```
* 4. Install all required packages from requirements.txt:
```
$ make install
```
* 5. After the installation process is complete, test the code in the Jupyter notebooks.

* 6. Open the corresponding Jupyter notebook, e.g., 1_Ridge_mode_use_2D_and_3D_feature_sets.ipynb.

* 7. Click 'Select Kernel' in the top-right corner of the notebook. Choose `.venv (Python 3.12.1) .venv/bin/python` as the kernel.

Now, you can start running the code in the Jupyter notebook!

# Prerequisites
This project requires the following dependencies:

1. **Java**
   - openjdk version "23.0.2"

2. **MATLAB**
   - Version: R2021b

# Compiling and Running the Java Code

If any changes are made to the Java program, follow these steps:

* 1. Open a terminal in the corresponding Java folder.
* 2. Compile the Java program using the following command:
```
$ javac -cp .:cdk-1.4.13.jar:cdk.jar:jmathio.jar:jmathplot.jar GetCDKDescriptors.java
```
* 3. Ensure that the `temp.sdf` file is present in the app/temp/ folder.
* 4. Run the Java program using the command:
```
$ java -cp .:cdk-1.4.13.jar:cdk.jar:jmathio.jar:jmathplot.jar GetCDKDescriptors temp
```
* 5. Verify that the Java code runs successfully.

