## Installation

### Requirements

- Linux
- CUDA 10.0
- Python 3.6+
- PyTorch 1.1
- Cython

### Create Environment ##
```shell
conda create -n propose_reduce python=3.6
conda activate propose_reduce
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
mkdir propose_reduce && cd propose_reduce
git clone https://github.com/dvlab-research/ProposeReduce.git
``` 
### Install MMDetection ###
```shell
cd ProposeReduce
pip install cython
./compile.sh
python setup.py install --user
```

### Install Libraries ###
```shell
cd libs/mmcv/
python setup.py develop
cd ../cocoapi/PythonAPI/
python setup.py develop
```
