# Online_neurSLS Documentation
Accompanying code for the paper "XXXXX".


# Online neurSLS Documentation

## Overview

This repository contains the code accompanying the paper titled "XX" authored by Danilo Saccani, Luca Furieri, and Giancarlo Ferrari Trecate.

For inquiries about the code, please contact:

- Danilo Saccani: [danilo.saccani@epfl.ch](mailto:danilo.saccani@epfl.ch)

## Repository Contents

1. **main.py**: Entry point for training the distributed operator using neural networks.
2. **utils.py**: Contains utility functions and main parameters for the codebase.
3. **models.py**: Defines models including the system's dynamical model, Recurrent Equilibrium Network (REN) model, and interconnection model of RENs.
4. **plots.py**: Includes functions for plotting and visualizing training and evaluation results.

## Getting Started

### Prerequisites

- Dependencies listed in `requirements.txt`

### Installation

1. Cloning the Repository

```bash
git clone https://github.com/DecodEPFL/Distributed_neurSLS.git
```

2. Navigate to the cloned directory:

```bash
cd Distributed_neurSLS
```
3. Install the required dependencies. We recommend using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Activate the virtual environment (Linux/macOS)
pip install -r requirements.txt
```

### Usage
1. Adjust parameters in utils.py as needed.
2. Run the main script to start training:
```bash
python main.py
```
## Example

### Coordination in mountains problem 

The following gifs show trajectories of the vehicles with the approach proposed in "Neural System Level Synthesis: Learning over All Stabilizing Policies for Nonlinear Systems" and the one with the proposed Online neurSLS approach.
# Mountains problem

<p align="center">
<img src="./GIF/mountains_pretrained_nodelta.gif" alt="robot_trajectories_with_neurSLS" width="400"/>
<img src="./GIF/mountains_onlineSLS_nodelta.gif" alt="robot_trajectories_with_online_neurSLS" width="400"/>
</p> 

# Mountains problem with $\delta$
<p align="center">
<img src="./GIF/mountains_pretrained_delta.gif" alt="robot_trajectories_with_neurSLS" width="400"/>
<img src="./GIF/mountains_onlineSLS_delta.gif" alt="robot_trajectories_with_online_neurSLS" width="400"/>
</p> 

# Dynamic obstacles problem
<p align="center">
<img src="./GIF/mountains_pretrained_delta_written.gif" alt="robot_trajectories_with_neurSLS" width="400"/>
<img src="./GIF/mountains_pretrained_delta_written.gif" alt="robot_trajectories_with_online_neurSLS" width="400"/>
</p> 

## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by] 

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg