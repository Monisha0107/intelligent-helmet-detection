
# Intelligent Helmet Detection using Machine Learning

A system designed to reduce road fatalities among two-wheeler riders by continuously monitoring helmet compliance while they are riding and actively ensures that riders wear helmets, promoting safer driving practices and helping to prevent accident-related deaths.


### Project Description

Once the engine turns on, the camera is placed in the cluster of the bike and starts to captures the rider's face regularly and detects whether the rider wears helmet or not, if there is no helmet, it follows certain constrains.

- Speed restricts to 30 kmph
- Alarm rings for 10 seconds
- Helmet LED blinking indicating the rider to wear helmet


### Workflow

First, it checks for the engine status, if it is in ON state, then the camera starts to capture the rider's video. If the rider wears a helmet, the function of the vehicle will be normal and it keeps checking for the rider's video. If the rider removes the helmet while driving, the vehicle automatically restricts the speed to 30 kmph gradually.


##  Getting Started

### Clone the Repository

```sh
git clone https://github.com/adityajai25/intelligent-helmet-detection.git
```
Then 

```sh
cd intelligent-helmet-detection
```


## Dataset Overview
The dataset is designed to train models to detect whether a rider is wearing a helmet or not. It contains labeled images categorized into two classes: **With Helmet** and **Without Helmet**. This dataset aims to enhance road safety by enabling systems that monitor helmet compliance among riders.

## Dataset Structure

The dataset is organized into the following classes:
- **With Helmet**: Images where the rider is wearing a helmet.
- **Without Helmet**: Images where the rider is not wearing a helmet.



## Creating Virtual Environment

Using a virtual environment isolates dependencies, manages library versions, keeps the global Python environment clean, and ensures consistent setups.

### On Windows

#### Creating a virtual environment:

Open Command Prompt and navigate to the project directory

```sh
cd project/directory/

```
Create a Virtual Environment:
```sh
python -m venv env
```
To Activate the Virtual Environment:

```sh
.\env\Scripts\activate
```

### On mac/Linux

#### Creating a virtual environment:
Open terminal and navigate to the project directory

```sh
cd project/directory/

```
Create a Virtual Environment:
```sh
python -m venv env
```
To Activate the Virtual Environment:

```sh
source env/bin/activate
```


## Installing Required Packages

Once the virtual environment is activated, install the required packages using the following commands:


#### 1. Install openCV-Python

```sh
pip install opencv-python==4.6.0.66
```
#### 2. Install numpy

```sh
pip install numpy==1.23.0
```

#### 3. Install tensorflow

```sh
pip install tensorflow==2.13.0
```


## Execution
After installing the packages required, the project needs to be executed in the following order.

The preprocessing is the process of preparing and cleaning the dataset to ensure it's in the optimal format for model training and its key functions include,

- Resizing and Scaling: 
    Adjusting image dimensions and scaling pixel values to standardize input data.
- Label Formatting: 
    Organizing and encoding labels to match the model's expected input format.

```sh
python preprocessing.py 
```

main.py script is the primary execution file that loads the trained model and initiates the helmet detection process

```sh
python main.py 
```


This will initiate the application, and it may take a few moments to activate the webcam and begin detection.
## Authors

- [ADITYA J P](https://www.github.com/adityajai25)
- [PREMSAIKUMAR S](https://www.github.com/prem1507)
- [RISHI T](https://www.github.com/rishithayanidhi)

