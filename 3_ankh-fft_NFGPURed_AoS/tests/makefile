#!/bin/bash

# Compilers
GCC      = g++ -O3 -ffast-math -mfma -fno-trapping-math -march=native -mtune=native -mavx -std=c++11 -I/usr/local/cuda/include
NVCC 	 = nvcc -O3 -std=c++11 -I/usr/local/cuda/include/

# Paths
MKLPATH      = /usr/lib/x86_64-linux-gnu/
MKLPATH_INC  = /usr/include/
FFTWPATH     = /usr/lib/x86_64-linux-gnu/

# Libraries and includes
INC_MKL  = -DMKL_ILP64 -m64 -I$(MKLPATH_INC) -Wl,--start-group $(MKLPATH)libmkl_intel_lp64.a $(MKLPATH)libmkl_sequential.a $(MKLPATH)libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
INCLUDE  = -lm -lpthread -lblas -llapack -L$(FFTWPATH)/lib -I$(FFTWPATH)/include -lfftw3 $(INC_MKL)

# CUDA-specific flags
CUDA_LIB_PATH = /usr/local/cuda/lib64
CUDA_INC_PATH = /usr/local/cuda/include
CUDA_LIBS     = -L$(CUDA_LIB_PATH) -lcudart

# Targets
all: analyze

# Check if short.cu exists
check:
	@if [ ! -f ../include/short_cuda.cu ]; then \
		echo "Error: short_cuda.cu not found!"; \
		exit 1; \
	fi


	
short_cuda.o: ../include/short_cuda.cu
	$(NVCC) $(DEBUG) -I$(CUDA_INC_PATH) -c ../include/short_cuda.cu -o short_cuda.o -maxrregcount=64 #--ptxas-options=-v 

analyze.o: analyze.cpp *.cpp ../include/*.hpp
	$(GCC) $(DEBUG) $(INCLUDE) -c analyze.cpp -o analyze.o

analyze: analyze.o short_cuda.o
	$(GCC) $(DEBUG) analyze.o short_cuda.o -o analyze $(INCLUDE) $(CUDA_LIBS)

clean:
	rm -f *.o analyze
