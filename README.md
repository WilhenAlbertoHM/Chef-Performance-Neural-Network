# Chefs-Performance-Neural-Network

## Overview
This project is part of the CS470 - Artificial Intelligence course here at the University of Massachusetts Boston, and it focuses on training a neural network to evaluate collaborative performance among 20 chefs. The neural network is designed to predict the outcomes of the number of complete/incomplete tasks, the average bonus <span>($)</span> each chef earned after the collaborative task, and the money <span>($)</span> that the company earned or lost after the task. 

## Requirements
* Python
* Keras
* NumPy
* scikit-learn

## Getting Started
If you want to learn more about neural networks and how they work, feel free to use this project for learning purposes. Here's how to get started.
1. Download and install Python - You can visit [Real Python](https://realpython.com/installing-python/) to learn how to install Python on Windows/macOS/Linux.
2. Clone the repository on your desired directory path - I recommend having this project on the Desktop or somewhere easily accessible.

```bash
cd Desktop
git clone https://github.com/WilhenAlbertoHM/Chef-Performance-Neural-Network/
cd Chef-Performance-Neural-Network
code .
```

Note: `code .` opens Visual Studio Code IDE. If you decide to use other IDEs, you can open the IDE of your choice and access the folder that way. 

3. Install the required libraries to run the program (assuming you have Python installed). The commands below can be written inside your IDE's terminal:

```bash
pip install numpy
pip install scikit-learn
pip install keras
```

4. Run the script:

To run `chef_performance_simple.py`:

```bash
python3 chef_performance_simple.py
```

To run `chef_performance_with_hidden_layers.py`:

```bash
python3 chef_performance_with_hidden_layers.py
```

## How to use
1. Load the dataset from `data/chef_data.pickle`. Feel free to transform it to a .csv file, using `pandas`. Otherwise, install `pickle` to load the data.
2. Build a neural network using Keras and scikit-learn.
3. Train the model using the training data for an arbitrary number of epochs, batch size, etc. You can explore different models and neural networks to see how they work and check which performs best.
4. Evaluate the model on the test data and print the results.

## Results and Analysis
After training the model, various metrics are calculated and printed:
* Mean and standard deviation of the differences between predictions and actual values.
* Training and testing values.
* Averages and standard deviations of the collaborative performance metrics.

### Simple Neural Network using a Linear Activation Function 
![image](https://github.com/WilhenAlbertoHM/Chef-Performance-Neural-Network/assets/92064680/623ccd43-51ea-457c-88c2-ab55aa08a304)

### More Complex Neural Network with Hidden Layers
![image](https://github.com/WilhenAlbertoHM/Chef-Performance-Neural-Network/assets/92064680/7cf97f7e-0188-499c-b9de-220892f69359)

## Future Improvements
For improvements, a graphical representation of the outcomes can be implemented for data visualization. This can be done with `Matplotlib` and `seaborn`. Also, other neural networks can reduce error even more, at the possible cost of complexity.

## Acknowledgements
This project comes from the CS470 - Artificial Intelligence course, at the University of Massachusetts Boston. This was published on GitHub for educational purposes regarding the topics of artificial intelligence and neural networks. 
