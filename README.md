A simple script to test TensorFlow GPU installation and configuration.
Run this script to see TensorFlow version, build info, and available GPU devices.
If you have a GPU and TensorFlow is configured correctly, you should see the GPU listed.
If you do not have a GPU, TensorFlow will run on the CPU.

Make sure to run this script in an environment where TensorFlow is installed.
When you install tensorflow (assuming you have a GPU) use one of these commands to install tensorflow with GPU support:

 - pip3 install tensorflow[and-cuda]
 - pip install tensorflow[and-cuda]

For more information, see https://www.tensorflow.org/install/pip

This script does not suppress any TensorFlow messages, so you can see any warnings or errors.
