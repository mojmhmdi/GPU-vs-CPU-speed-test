
# run time comparison gpu vs cpu
from ast import Return
import os
import torch
import time
import tensorflow as tf
# for cpu, uncomment lines 9, restart your kernel and run the code again
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# check wether kernel detects your GPU
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print('GPU is detected')
else:
    print('GPU is not detected')


a = tf.random.Generator.from_seed(42)
al = time.time()
for i in range(150):
    a.normal(shape=(100000, 200))
b = time.time()
print('your run time is: %s' % (b-al))
