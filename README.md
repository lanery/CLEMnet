# Deep CLEM
Convolutional neural network for prediction of fluorescence in EM images


#### Setup instructions for hpc29

* Install miniconda
```
[rlane@hpc29:~]$ bash ./Miniconda3-latest-Linux-x86_64.sh
```
* Create python environment
```
[rlane@hpc29:~]$ conda create -n fmml_gpu tensorflow-gpu pandas numpy scikit-learn scikit-image opencv seaborn shapely jupyterlab beautifulsoup4 hdf5 matplotlib scipy altair vega_datasets
[rlane@hpc29:~]$ conda activate fmml_gpu
(fmml_gpu) [rlane@hpc29:~]$ git clone https://www.github.com/fcollman/render-python
(fmml_gpu) [rlane@hpc29:~]$ cd ./render-python/
(fmml_gpu) [rlane@hpc29:render-python]$ python ./setup.py install
(fmml_gpu) [rlane@hpc29:~]$ pip install --no-cache-dir git+https://github.com/lanery/iCAT-workflow
```


#### To run in a Jupyter notebook
```
[rlane.TUD278418] ➤ ssh -L localhost:8893:localhost:8892 rlane@hpc29
(base) [rlane@hpc29:~]$ conda activate fmml_gpu
(fmml_gpu) [rlane@hpc29:~]$ jupyter lab --no-browser --port=8892
```
