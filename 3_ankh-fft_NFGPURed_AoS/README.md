## ANKH
The ankh-fft code is a c++ implementation of the ANKH-FFT method presented in https://arxiv.org/abs/2212.08284 . This library aims at efficiently perform energy computation arising in molecular dynamics.

# Dependencies
The ankh-fft code depends on both BLAS and FFTW3

# Install and test
In the root directory, change the BLAS and FFTW3 paths in ./tests/makefile
then run

$ cd ./tests

$ make analyze

$ ./analyze 2 4 15 6 5

Signification of the various parameters can be found in analyze.cpp (i.e. equispaced interpolation order / number of tree levels / number of far images in each direction / interpolation order for far iamges / Chebyshev interpolation order at leaves).

# Licence
The ankh library is under Lesser Gnu Public Licence 3 (LGPL).

# Author
Igor Chollet (Laboratoire Analyse Geometrie et Applications / Laboratoire de Chimie Theorique)


# CUDA codes
a naive gpu implementation is presented using CUDA-C. 
All data for each box (its particles and potentials) are stored in one array near together (like AoS scheme) and data for all boxes are stacked.
in short_cuda.cu:
	short_range_naiiveGPU_dataCollection 
		restructures dataset
	short_range_wrapper and short_range_cuda 
		compute p2p
		each thread handles a single last level box
		
