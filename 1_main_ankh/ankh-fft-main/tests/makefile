#!/bin/bash

GCC      = g++ -O3 -ffast-math -mfma -fno-trapping-math -march=native -mtune=native -mavx -std=c++11
MKLPATH  = /usr/lib/x86_64-linux-gnu/
MKLPATH_INC = /usr/include/ 
FFTWPATH = /usr/lib/x86_64-linux-gnu/
INC_MKL  = -DMKL_ILP64 -m64 -I$(MKLPATH_INC) -Wl,--start-group $(MKLPATH)libmkl_intel_lp64.a $(MKLPATH)libmkl_sequential.a $(MKLPATH)libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
INCLUDE  = -lm -lpthread -lblas -llapack -L$(FFTWPATH)/lib -I$(FFTWPATH)/include -lfftw3 $(INC_MKL)

analyze : analyze.o makefile
	$(GCC) $(DEBUG) analyze.o -o analyze $(INCLUDE)

analyze.o: analyze.cpp *.cpp ../include/*.hpp
	$(GCC) $(DEBUG) $(INCLUDE) -c analyze.cpp -o analyze.o
