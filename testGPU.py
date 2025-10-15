"""
A simple script to test TensorFlow GPU installation and configuration.
Run this script to see TensorFlow version, build info, and available GPU devices.

Make sure to run this script in an environment where TensorFlow is installed.
When you install tensorflow have a GPU and want to use it use pip3 install tensorflow[and-cuda]
or pip install tensorflow[and-cuda]
For more information, see https://www.tensorflow.org/install/pip

If you have a GPU and TensorFlow is configured correctly, you should see the GPU listed.
If you do not have a GPU, TensorFlow will run on the CPU.
This script does not suppress any TensorFlow messages, so you can see any warnings or errors.
"""

import tensorflow as tf
print("VERSION:", tf.__version__)
print("BUILD INFO:", tf.sysconfig.get_build_info())
print("PHYSICAL DEVICES:", tf.config.list_physical_devices('GPU'))
print("LOGICAL DEVICES:", tf.config.list_logical_devices('GPU'))

print("NUM GPUs AVAILABLE:", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU")
else:
    print("TensorFlow is using the CPU")
# Test if TensorFlow can access the GPU
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    print("TensorFlow successfully performed a matrix multiplication on the GPU.")
    print("Result:\n", c.numpy())
except RuntimeError as e:
    print("Error: TensorFlow could not perform a matrix multiplication on the GPU.")
    print(e)