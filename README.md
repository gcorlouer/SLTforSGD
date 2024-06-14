# SGD on degenerate 1D and 2D loss-landscape

This project contains python code to to study the influence of degenerate directions in the loss landscape on SGD dynamics.
SGD is ran on 1D and 2D models that are linear in their inputs and polynomial in their parameters. 

The main classes and functions are in the in the SGD_utils.py file in the library lib/ directory 

## Codebase structure

Folders
* `lib/` folder contains `SGD_utils.py` the main library of the project to run SGD on degenerate models
* `notebooks/` contains notebook showing SGD escaping non degenerate minima and getting stuck on degenerate minima

## Usage

Install dependencies with `requirement.txt` and run experiments in notebooks.




