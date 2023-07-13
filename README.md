# resnetrk3568
resnet for rk3568

# Use
- ubuntu 22.04
- python resnet18.py
- create resnet18.onnx
- create data
- python rknn_transfer.py
- create resnet_18.rknn
- python cifar10_to_jpg.py
- data to Data jpg files
- 
- debain 10 in rk3568
- https://github.com/rockchip-linux/rknpu2/blob/master/runtime/RK356X/Linux/librknn_api/aarch64/librknnrt.so
- scp librknnrt.so /usr/lib
- https://github.com/rockchip-linux/rknn-toolkit2/blob/master/rknn_toolkit_lite2/packages/rknn_toolkit_lite2-1.5.0-cp37-cp37m-linux_aarch64.whl
- pip install *.whl
- scp Data to debain
- python rknnlite_inference0.py
- python rknnlite_inference1.py

