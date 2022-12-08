# WalkingDropletTracking
This repository includes all the datasets, codes, and other resources discussed in our paper X. 


Here is a brief overview of the content of this repo:


1- datasets folder includes all the walking droplet and granular flow experiments with original videos and annotations etc.


2- train_results has the metrics regading the model training/testing process. 


3- tracking_results has all the real-time tracking videos as well as csv files generated out of these experiments. We highly recommend to have a look at some of them.


4- figures folder should have all the figures we generated in the paper. paper_figures.ipynb generates some of these figures. 


5- comparision folder has all the materials regarding the comparision between Hungarian Algorithm and StrongSORT.


6- yolov5, Yolov5_StrongSORT_OSNet and labelImg are repos cloned from their original sources. For uniformity, we added them to our repo. Inside Yolov5_StrongSORT_OSNet/sort_track_results folder, 
you can find all tracking videos carried out with StrongSORT. Pay attention to multiple ID switches.

7- best_droplet.pt and best_intruder.pt are the Pytorch YOLO models we trained for walking droplet and granular flow experiments, respectively. 


We provide a complete training course for your specific problem domain in **tutorial.ipynb**. This will walk you through all the steps from collecting the data to training your own tracker. All of the main driver functions are located in **myutils.py**. To get started, first of all install conda to your system, link is  [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). I tested everything on my linux machine with Ubuntu 22.04.1 LTS. To create the conda environment, do the following from your terminal;

> conda env create -f environment.yml


> conda activate droptracker


> python3 -m pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip


> jupyter notebook

After that, you can go ahead and start working on **tutorial.ipynb** to create your own tracker. I did my best to explain every step in details. I hope you find it useful. Please let us know if you need any question or help. You can shoot an email to erdikara@spelman.edu.
