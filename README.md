# Physics informed learning of phase field equations
This repository contains codes for physics informed learning of phase field equations. One dimensional examples of allen cahn and cahn hilliard equations are presented here. Unlike the traditional approach of PINN's where the parameters of the equation are learned from gradient descent, here the parameters of the equation are approximated using least squares fit at every training epoch. For the purpose of comparison the codes where the equation parameters are learned by traditional gradient descent are also presented.


## Getting started
The first step is to install anaconda in your system
and create the conda enviroment necessary to run the above codes.

```
conda create env -f sciml_gpu.yaml
cond activate sciml_gpu
```
codes where least squares fit is used to learn equation parameters
```
python PINN_AC_1D_LSfit.py
python PINN_CH_1D_LSfit.py
```
codes where gradient descent is used to learn equation parameters

```
python PINN_AC_1D.py
python PINN_CH_1D.py
```

All the plots and data files will be automatically stored in the corresponsing results folder.



# AC Equation Solver

This Python script solves the AC equation using the Physics-Informed Neural Networks (PINN) approach. It utilizes the SIREN (Sinusoidal Representation Networks) architecture for modeling the primary and auxiliary variables in the equation.

## Requirements

- Python 3.x
- NumPy
- SciPy
- PyTorch
- Matplotlib
- tqdm

## Installation

1. Clone the repository or download the source code files.
2. Install the required dependencies using the following command:
   ```
   pip install numpy scipy torch matplotlib tqdm
   ```

## Usage

1. Ensure that the `data/AC.mat` file is present in the same directory as the script. This file contains the input data for the AC equation.
2. Open the script `ac_equation_solver.py` in a Python editor or IDE.
3. Modify the script parameters and settings as per your requirements, such as learning rates, hidden layer sizes, batch size, number of epochs, etc.
4. Run the script using the following command:
   ```
   python ac_equation_solver.py
   ```

## Output

The script will perform the training process and save the following files in the `result_AC_1d` directory:

- `U_data_<params>.npy`: Numpy array containing the original, predicted, noisy, and error values of the primary variable phi.
- `Loss_collect_<params>.npy`: Numpy array containing the loss values at each epoch during training.
- `params_collect_<params>.npy`: Numpy array containing the learned parameter values at each epoch during training.
- `model_u_<params>.pt`: Saved PyTorch model of the primary SIREN network.
- `<plots>.png`: Plots of the original, noisy, and predicted solutions, as well as the loss curve.
