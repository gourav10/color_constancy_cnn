# Color Constancy using CNN

### Project Author
	1. Heet Sakaria
	2. Gourav Beura

### Wiki Link
https://wiki.khoury.northeastern.edu/x/aiKiBw

### Operation System
Windows 11

### IDE
Jupyter Notebook

### Instructions for installing dependencies the python files
	Use the command 
    ```
    $ pip install -r requirements.txt
    ```

    to install the dependencies.
	
### Instructions to setup the dataset
	1. If you have downloaded Shi-Gehler dataset, you will need to process these HDR images, using `preapare_dataset.py`. At the end of the file, replace the `path` variable with the directory to the **Shi_Gehler folder**. For example, `path = ...//Dataset//Shi_Gehler` and inside the **Shi_Gehler folder** you will need to create the subfolders which look like [this](https://imgur.com/a/tIfyEMp) and inside the **gt folder** you put the ground truth illuminant files and it looks like [this](https://imgur.com/a/CJ1ELtP). After that, you are ready to go.
    2. First, you can create your own train and test set by running the script 
    ```
    generate_data.py
    ```
    You need to have the 'color-casted' images and corresponding ground truth illuminant matrix. We work with the Shi-Gehler dataset so these 'color-casted' images are the processed images from the original [HDR images](http://www.cs.sfu.ca/~colour/data/shi_gehler/) and the corresponding ground truth illuminant [here](http://www.cs.sfu.ca/~colour/data/shi_gehler/groundtruth_568.zip). If you have already processed the HDR images, you would have about 500+ 'color-casted' images. You will need to divide these image into two parts.
	3. Then, inside the `generate_data.py`, within the function `generate_train_data`, replace the `path` variable with the directory to your train set, for example: `path = 'C:\\Users\\...\\Shi_Gehler\\Train_set\\'`, and do the same for the function `generate_test_data` with the directory of your test set.
	3. Open the  **color_constancy_playbook.ipynb** files in Jupyter Notebook and run each cell.
	
### Project Repo:
	https://github.com/gourav10/Monocular_Depth_Estimation
	
### References
    * Solving the color constancy porblem with CNN model proposed by [Simone_et_al](https://arxiv.org/pdf/1504.04548.pdf)
    * https://github.com/hidiryuzuguzel/cc-cnn 
    * https://github.com/WhuEven/CNN_model_ColorConstancy