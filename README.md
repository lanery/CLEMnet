# CLEMnet
Convolutional neural network for prediction of fluorescence in EM images

Created by Luuk Balkenende, Lennard Voortman, and Ryan Lane

### Clone from GitHub
```
git clone https://github.com/lanery/CLEMnet.git
```
or install via pip
```
pip install git+git://github.com/lanery/CLEMnet.git
```

### Setup instructions for hpc29
* Install miniconda
```
[rlane@hpc29:~]$ bash ./Miniconda3-latest-Linux-x86_64.sh
```
* Create python environment
```
[rlane@hpc29:~]$ conda create -n fmml_gpu tensorflow-gpu numpy scipy scikit-image jupyterlab matplotlib
[rlane@hpc29:~]$ conda activate fmml_gpu
```

Other useful python packages
```
scikit-learn opencv pandas
```

### To run in a Jupyter notebook
* Start jupyter lab session
```
[rlane.TUD278418] ➤ ssh -L localhost:8893:localhost:8892 rlane@hpc29
(base) [rlane@hpc29:~]$ conda activate fmml_gpu
(fmml_gpu) [rlane@hpc29:~]$ jupyter lab --no-browser --port=8892
```
* Navigate to http://localhost:8893/lab and enter token
* (Optional) environment setup for using CATMAID viewer
```
(fmml_gpu) [rlane@hpc29:~]$ conda install -c conda-forge ipywidgets ipympl nodejs
(fmml_gpu) [rlane@hpc29:~]$ jupyter nbextension enable --py widgetsnbextension
(fmml_gpu) [rlane@hpc29:~]$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
* Then restart jupyter lab

### (Optional) `iCAT-workflow` installation
```
(fmml_gpu) [rlane@hpc29:~]$ git clone https://www.github.com/fcollman/render-python
(fmml_gpu) [rlane@hpc29:~]$ cd ./render-python/
(fmml_gpu) [rlane@hpc29:render-python]$ python ./setup.py install
(fmml_gpu) [rlane@hpc29:~]$ pip install --no-cache-dir git+https://github.com/lanery/iCAT-workflow
```
