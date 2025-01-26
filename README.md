## ANKH
The ankh-fft code is a c++ implementation of the ANKH-FFT method presented in https://arxiv.org/abs/2212.08284 . This library aims at efficiently perform energy computation arising in molecular dynamics.


## redundancy
The main (under-developing) idea for redundacy technique for P2P computations is presented in https://arxiv.org/abs/2403.01596 . Here, a naive GPU impementaion for P2P calculation is presented, then a redundancy technique is applied. In each directory, a different datastructur is used, and results are stored in excel files (*.odb). 
Overllay, ANKH is efficient for far-field computations but for near-field it was hard to achieve speedup on GPU, that is mostly beacuse of its energy functions that makes device register pressure problem. Since improving energy function was not subject of our work, we only focused on data structure and left kernel functions imporvement for future works.
All tests were done on UBUNTU 24, Core i7 7th Gen, 16GB RAM, NVIDIA GeForce GTX1050

