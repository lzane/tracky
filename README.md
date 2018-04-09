# Tracky 基于深度全卷积神经网络的实时物体识别与追踪

## Dependencies
Python3, tensorflow 1.4, numpy, opencv 3, Cython, sacred, matplotlib, pillow, scipy

## Getting started
clone the repository
```
git clone --recurse-submodules -j8 https://github.com/lzane/tracky.git
cd tracky
```

Build the Cython module
```
cd darkflow
python3 setup.py build_ext --inplace
```

Download the model and config file, you can find the link inside the Submodules

Modify the `tracky.py` so that it point to the correctly path of your model and config file.

Run the tracky by
```
python3 tracky.py
```






