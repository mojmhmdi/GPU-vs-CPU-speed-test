
# run time of gpu
import torch
import time
import os
# for cpu,uncomment line 8, comment line 18 and uncomment line 19

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# check wether kernel detects your GPU
torch.cuda.is_available()

cuda = torch.device('cuda')  # Default CUDA device
al = time.time()
for i in range(150):
    # for cpu just delet device =  cuda
    torch.randn((100000, 4000), device=cuda)  # for GPU
    #torch.randn((100000, 200))  # for CPU

b = time.time()
print(f'your run time is: {b-al}', 's')
