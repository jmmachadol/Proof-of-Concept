# Evaluation of a Neural Network for the Solution of Differential Equations

This repository contains the code and resources for the undergraduate thesis project "Evaluation of a Neural Network for the Solution of Differential Equations". The project aims to explore the use of neural networks as an alternative method for solving partial differential equations (PDEs), with a focus on the Poisson equation in one dimension.


## Overview

The main goal of this project is to develop and evaluate a proof of concept using a Multi-Layer Perceptron (MLP) regressor to solve the Poisson equation in 1D. The objectives of this project are:

1. Document the state of the art for neural networks as an approximation method for solving differential equations.
2. Propose a neural network as a solution method for the Poisson differential equation that preserves the input/output of traditional algorithms.
3. Evaluate the solution capabilities of the neural network through a computational experiment design.
4. Adjust the parameters of the network to reproduce the traditional scheme of a numerical method in terms of input-output.
5. Compare the results obtained by the neural network with traditional methods.

In addressing these objectives, the following aspects will be considered:

1. Interdisciplinary approach: Combining knowledge from physics, mathematics, and computer science to address the problem.
2. Flexibility: The proposed neural network should be adaptable to different boundary conditions, geometries, and materials.
3. Optimization: The neural network should be tunable to optimize its performance and accuracy.
4. Scalability: The approach should be extendable to other types of PDEs and higher dimensions.


## Structure

The repository is organized into the following directories and files:

- `src/` : Contains the source code scripts.
    - `propuesta_redneuronal.py`: Proposal of the neural network for the resolution of the Poisson equation.
    - `visualizacion_exploratoria.py`: Visualization of the network's performance.
    - `bayessearch.py`: Hyperparameter optimization for the neural network using Bayesian search.
    - `gridsearch.py`: Hyperparameter optimization for the neural network using grid search.
    - `randomizedsearch.py`: Hyperparameter optimization for the neural network using randomized search.
    - `fem_1D.py`: Comparison of the network's performance with finite element method in 1D.
    - `cross_validation.py`: Implementation of cross-validation to assess model performance.
    - `variado_conjuntos.py`: Variation of the number of data for training the neural network.
- `results/`: Directory for storing the results of the experiments.
- `docs/`: Contains additional documentation and notes related to the project.
- `LICENSE`: Project license file.
- `README.md`: User guide for understanding and navigating the project.
- `requirements.yml`: List of dependencies needed to reproduce the coding environment.


## Dependencies

This project requires Python 3.7+ and the following libraries:

- NumPy
- SciPy
- scikit-learn
- Matplotlib

The specific versions of these libraries used in the project can be found in the `requirements.yml` file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments

I would like to thank my thesis advisor, Dr. Nicolás Guarín-Zapata (@nicoguaro), for their guidance and support throughout this project.
