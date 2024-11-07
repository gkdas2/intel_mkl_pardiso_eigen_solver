A simple interface to Intel MKL Pardiso solver to solve sparse linear systems of form Ax=b, where A and b are Eigen data structures

# Requisites
- cmake
- Eigen
- Intel MKL Library

# How to setup
Use CMakeListst.txt to see cmake flags for Intel MKL

# How to use 
See pardiso/test.cpp for implementation
You may need to set MKL environment variables first, if not already done. Most likely:
> ``` 
> source /opt/intel/oneapi/setvars.sh
>```

To compile the example:
> ``` 
> git clone https://github.com/gkdas2/intel_mkl_pardiso_eigen_solver.git
> cd intel_mkl_pardiso_eigen_solver
> mkdir build; cd build 
> cmake .. 
> make 
> export OMP_NUM_THREADS=4 
> ./pardiso_solver 
>```



