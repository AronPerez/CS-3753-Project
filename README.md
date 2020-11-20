# CS-3753-Project
CS-3753-002-Fall-2020-Data Science

# Synopsis
The L5Kit data is stored in zarr format which is basically a set of numpy structured arrays. Conceptually, it is similar to a set of CSV files with records and different columns. As for any zarr file, there must be a root folder. So, the L5Kit data consists of groups of zarr datasets, we can see it by using zl5.info

# Data Overview

Scenes : a collection of frames:

- Frames : a scene has a list of frames that start from scene.frame_index_interval[0] and ends at scene.frame_index_interval[1]
- Host : a scene has a host which is the AV that films the scene.
- Timestamps: a scene has a start_time and an end_time

Frames : a collection of agents (the host agents + other agents)
Agents : Any object in circulation with the automatic vehicle (AV)
agents_mask: a mask that (for train and validation) masks out objects that aren't useful for training. In test, the mask (provided in files as mask.npz) masks out any test object for which predictions are NOT required.
Traffic_light_faces : traffic lights and their faces (bulbs)

L5Kit
===============================
- Load driving scenes from zarr files
- Read semantic maps
- Read aerial maps
- Create birds-eye-view (BEV) images which represent a scene around an AV or another vehicle
- Sample data
- Train neural networks
- Visualize results


First-Time Setup & Access
===============================
1. SSH into your account
2. Create a directory you're gonna work in (inside your home) and cd into it
3. Within the project directory, we’ll create a Python virtual environment. For the purpose of this tutorial, we’ll call it my_project_env but you should call it something that is relevant to your project.
virtualenv my_project_env
4. Before we install Jupyter, we need to activate the virtual environment. You can do that by typing: source my_project_env/bin/activate
Your prompt should change to indicate that you are now operating within a Python virtual environment. Your command prompt will now read something like this: (my_project_env)user@host:~/my_project_dir$. At this point, you’re ready to install Jupyter into this virtual environment.
5. pip install jupyter
Note: Change XXXX to the port of your choice. Usually, the default is 8888. 
You can try 8889 or 8890 as well.
In your remote, the notebook is now running at the port XXXX that you specified. What you’ll do next is forward this to port YYYY of your machine so that you can listen and run it from your browser. To achieve this, we write the following command:
6. ON LOCAL RUN THE FOLLOWING
ssh -i <nameOfPrivateKey> -L <portYouChose>:localhost:<portYouChose> <username>@142.93.63.33
7. source <my_project_env>/bin/activate
8. jupyter notebook
9. You should see something like this https://gyazo.com/33c6409d5b21ec22d8f0b1c931a7ab3c
10. Open this on your local machine and boom, we're connected to the dataset
Source: https://www.digitalocean.com/community/tutorials/how-to-set-up-jupyter-notebook-with-python-3-on-ubuntu-20-04-and-connect-via-ssh-tunneling
11. Follow https://www.kaggle.com/kneroma/zarr-files-and-l5kit-data-for-dummies for setup of env
