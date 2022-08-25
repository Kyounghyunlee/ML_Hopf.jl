# ML_Hopf

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Julia package to simulate numerical examples and physical example of paper "Modelling of physical systems with a Hopf bifurcation using mechanistic models and machine learning"

Tested for julia versions 1.6 and 1.7.

## Installation
Pkg.clone("https://github.com/Kyounghyunlee/ML_Hopf.jl")

## Numerical examples
1. Van der Pol oscillator: all the figures are generated from the file vdp_ex.jl
2. Aeoroelastic model: all the figures are generated from the file num_ex.jl

## Experimental demonstration
1. The model can be trained using phys_experiment.jl
2. Validation (Fig.10) can be plotted using phys_ex_LOOCV_all.jl
3.  For the simulation of Fig.10, use phys_ex_LOOCV_1~4.jl