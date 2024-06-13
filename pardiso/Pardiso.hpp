/**
*Copyright (c) 2024 Ghanendra K Das
* MIT License 
*/


#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mkl_pardiso.h>

/**
 * @brief A direct linear solver for Ax=b using Intel MKL Pardiso.
 */
class Pardiso
{
public:
    enum MatrixTypes
    {
        SymmetricReal = 1,
        SymmetricRelPositiveDefinite = 2,
        SymmetricRealIndefinite = -2,
        SymmetricComplex = 3,
        HermitianComplexIndefinite = -4,
        HermitianComplex = 6,
        UnsymmetricReal = 11,
        UnsymmetricComplex = 13
    };

private:
    enum Phases
    {
        Analysis = 11,
        Analysis_NumericalFactorization = 12,
        Analysis_Factorization,
        Solve = 13,
        Factorization = 22,
        Factorization_Solve = 23,
        Solve_IterativeRefinement = 33,
        OnlyForwardSubstititution33 = 331,
        OnlyDiagonalSubstititution33 = 332,
        OnlyBackwardSubstititution33 = 333,
        RealeaseInternalMemoryLU = 0,
        ReleaseAllInternalMemory = -1,
    };

public:
    /**
     * @brief Construct a new Pardiso:: Pardiso object.
     *
     * @param matrixType : Describe the nature of the matrix on LHS. see Pardiso::MatrixType for different options. Default is real unsymmetric.
     */
    Pardiso(MatrixTypes matrixType = MatrixTypes::UnsymmetricReal);

    ~Pardiso();

    /**
     * @brief Factorize the matrix on LHS and store the factorization.
     *
     * @param A Input LHS sparse matrix
     */
    void Factorize(Eigen::SparseMatrix<double, Eigen::RowMajor> &A);

    /**
     * @brief Solve Ax = b and store in x. Assumes the factorization is complete.
     * Use this to solve for x for different b by reusing factorization of A previously done.
     *
     * @param b
     * @param x
     */
    void SolveSystem(Eigen::VectorXd &b, Eigen::VectorXd &x);

    /**
     * @brief Solve a linear system Ax=b. All steps are performed.
     *
     * @param A LHS sparse matrix
     * @param b RHS vector
     * @param x solution vector
     */
    void SolveSystem(Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::VectorXd &b, Eigen::VectorXd &x);

    /**
     * @brief Free up memory and end solver data
     *
     */
    void Terminate();

private:
    void *pt[64] = {0}; // Pardiso internal data. Also stores factorization so that we can reuse it
    MKL_INT iparm[64];  // Pardiso parameters
    MKL_INT maxfct = 1;
    MKL_INT mnum = 1;                             // number of RHS
    MKL_INT phase = Phases::Analysis;             // which staege in the solution process: factorization, solution, etc
    MKL_INT mtype = MatrixTypes::UnsymmetricReal; // Tells whether matrix is symmetric, positive definite etc. Default is unsymmetric real
    MKL_INT error;
    MKL_INT msglvl;
    MKL_INT *perm = nullptr;
    MKL_INT n;        // Number of equations, numrows in A
    MKL_INT nrhs = 1; // Number of right-hand sides
    MKL_INT idum;     // Dummy variable of int type
    double ddum;      // Dummy variable of double type

    bool IsFactorized = false;
    bool IsMemoryFreed = false;
};
