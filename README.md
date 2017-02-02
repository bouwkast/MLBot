# Machine Learning Halite Bot using TensorFlow

This project will attempt to explain how to setup a machine learning environment that can train a Halite bot based off of some replay files.

The goal of this is not to explore the formulas and math that are the foundation of machine learning, but to set up a user-friendly environment that is applicable to many problems that can be solved through the use of machine learning.

While this will focus largely on the programming challeng/game [Halite](halite.io), 
it should be a general enough approach that you will be able to apply the methods here
 quite easily and gain a better understanding for the mechanics behind machine learning.


Credit to Github user [brianvanleeuwen](https://github.com/brianvanleeuwen) for supplying the starter code for this machine
learning bot. - [repo link](https://github.com/brianvanleeuwen/Halite-ML-starter-bot)


### Prerequisites



There are several things that must be setup before you can run this project.
**Graphics Card**
The first of which will be taking note what kind of graphics card you have - if you have an
Nvidia based GPU you're in luck because you'll be able to use the GPU-based version of TensorFlow,
it is much quicker than the base CPU version - unless you have a fancy server CPU.

The benefit of having a GPU is simply the increase in speed, but this is entirely doable
with a CPU.

**Windows**

I"ll be making the assumption that you are on a recent iteration of the Windows operating system.
Some of the setup is much easier on Linux and OSX, but Windows works just as well and will
be referenced throughout the rest of this.


**Anaconda**
Downloading and installing the Anaconda python environment is the quickest and easiest way to 
get started with scientific research using Python because it comes with many core packages
that are annoying to install by themselves. 

[Download Anaconda](https://www.continuum.io/downloads)

When installing I would recommend **not** choosing the option to install for all users. If
you do choose to install for all users you must run Anaconda as admin, which is annoying.


**Nvidia**

There are two things we need from Nvidia: [Cuda Toolkit 8.0](https://developer.nvidia.com/cuda-toolkit) 
and [cudNN v5.1](https://developer.nvidia.com/cudnn)
Follow the installation instructions for both exactly - install the toolkit and then cudNN
**Don't install these if you don't have an Nvidia GPU.**


### Installing Keras and TensorFlow

It may be to your benefit to create an [Anaconda environment](https://www.tensorflow.org/get_started/os_setup#anaconda_installation)
 to do this, I won't cover that though.

Open up an Anaconda prompt - if you've installed Anaconda for all users run it as admin.

Type in the following:


```
pip install keras
```

This will install the [Keras](https://keras.io/) package, which we will use to sit on top of TensorFlow.

If it prompts you to do anything (y/n) choose y

Next we need to install [TensorFlow](https://www.tensorflow.org/get_started/os_setup)

In the same Anaconda prompt type in the following if you want the GPU-based TensorFlow package

```
pip install tensorflow-gpu
```


If you don't want the GPU-based TensorFlow package type the following:
```
pip install tensorflow
```


If you ever need to uninstall them replace "install" with "uninstall"


You'll want to test to ensure that everything is setup properly.
To do this, make sure that you have the repository cloned and run the following 
in a command prompt(or batch file):

```
python train_bot.py ./minireplays
```

This should load up around 50 replay files and begin running through the several epochs
building a model that you will be able to run your bot off of.

## Fixing Errors

If you are using the GPU-based version of TensorFlow, you might have ran into an issue
when you ran the above line that several CUDA libraries failed to load.

This most likely means you don't have the cudNN folder in your PATH variable and/or the files
within the cudNN folder are not being found by the CUDA Toolkit.

To guarantee that it will work add the cudNN folder to your PATH and add the files
within it to your CUDA Toolkit directory.

This _should_ solve any errors - otherwise look on the 
[TensorFlow help site](https://www.tensorflow.org/get_started/os_setup#common_problems)


## Submitting the Bot

When you have a model that you created add the following to a directory:
hlt.py
MyBot.py
networking.py
model.m5

## Further Inspection

Please look to the comments found in train_bot.py for information - this will be updated later


