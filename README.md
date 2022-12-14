# WalkingDropletTracking
This repository includes all the datasets, codes, and other resources discussed in our paper X regarding walking droplets and granular flow experinents. 


We provide a complete training course for your specific problem domain in **tutorial.ipynb**. This will walk you through all the steps from creating the dataset to training your own tracker. I tested the following steps on my linux machine with Ubuntu 22.04.1 LTS.To get started, first install conda to your system, link is  [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), then create/activate your environment using;


>> conda env create -f environment.yml


>> conda activate droptracker


>> python -m pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip



We would like to store our dataset, tracking results, and StrongSORT tracking results externally, as their combined size is approximately 8GB. To do this, use the following script to download and unzip the necessary files, and create three folders named *dataset*, *tracking_results*, and *sort_track_results* in your current directory.


>> python download_data.py


Lastly, start a notebook and start working on **tutorial.ipynb** to create your own tracker.All of the main driver functions are located in **myutils.py**. I did my best to explain every step in details. I hope you find it useful. 



>> jupyter notebook



Explantions of some of the files/folders are as follows;



1- *datasets*: all the walking droplet and granular flow experiments with original videos and annotations etc.


2- *train_results*: major results regading the model training/testing process. 


3- *tracking_results*:  real-time tracking videos as well as csv files generated out of these experiments. We highly recommend to have a look at some of them.


4- *figures folder*:  all the figures generated in the paper. paper_figures.ipynb generates some of these figures. 


5- *comparison folder*:  materials regarding the comparision between Hungarian Algorithm and StrongSORT.


6- *sort_track_results*:  tracking videos carried out with StrongSORT. Pay attention to multiple ID switches.


7- yolov5, Yolov5_StrongSORT_OSNet and labelImg are repos cloned from their original sources. For uniformity, we added them to our repo


8- best_droplet.pt and best_intruder.pt are the Pytorch YOLO models we trained for walking droplet and granular flow experiments, respectively. 




Please let us know if you need any question or help. You can shoot an email to erdikara@spelman.edu.
