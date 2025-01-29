### Prerequisites
This project has the following dependencies:

1. **Java**
   - openjdk version "23.0.2"

2. **MATLAB**
   - Version: R2021b


If any changes were made to java program. Using Termianl in the corresponding java folder, perform:
javac -cp .:cdk-1.4.13.jar:cdk.jar:jmathio.jar:jmathplot.jar GetCDKDescriptors.java

Mke sure there are temp.sdf file in the folder app/temp/, then perform:
java -cp .:cdk-1.4.13.jar:cdk.jar:jmathio.jar:jmathplot.jar GetCDKDescriptors temp 

to ensuere the java code can work.


Recommended Running Environment: Github Code Space
Instructions:
After fork the repository, open github code space by click 'Code' --> 'Code Spaces' --> 'Create Code Space on main'

After entering the code space, first create a virtual environment by runing following commnd in TERMINAL.
$ python -m venv .venv
$ source .venv/bin/activate

Then install all packages in requirements.txt by runing
$ make install 

After the install process finish, we can test code in the notebooks
Enter corresponing jupyter notebook, for example '1_Ridge_mode_use_2D_and_3D_feature_sets.ipynb'

Click 'select kernel' on the right corner of the notebook. Then choose '.venv(Python 3.12.1) .venv/bin/python'

Then we can start runing the code in the jupyternotebook