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

