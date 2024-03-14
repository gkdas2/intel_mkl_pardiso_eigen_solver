#include "Pardiso.hpp"

Pardiso::Pardiso(MatrixTypes matrixType)
{
    // Bunch of defaults. We will change later according to input.
    // https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/ssr/ssr_pardiso_parameters.htm

    mtype = matrixType;

    for (int i = 0; i < 64; i++)
    {
        iparm[i] = 0;
    }
    iparm[0] = 1;  // No solver default
    iparm[1] = 2;  // Fill-in reordering from METIS
    iparm[3] = 0;  // No iterative-direct algorithm
    iparm[4] = 0;  // No user fill-in reducing permutation
    iparm[5] = 0;  // Write solution into x
    iparm[9] = 13; // Perturb the pivot elements with 1E-13
    iparm[10] = 1; // Use nonsymmetric permutation and scaling MPS
    iparm[12] = 0; // Maximum weighted matching algorithm is switched-off
    iparm[13] = 0; // Output: Number of perturbed pivots; suppressed at 0
    iparm[17] = 0; //-1; // Output: Number of nonzeros in the factor LU; suppressed at 0
    iparm[18] = 0; //-1; // Output: Mflops for LU factorization; suppressed at 0
    iparm[26] = 1; // Check for zero or negative pivots
    iparm[34] = 1; // zero-based
    iparm[36] = 0; // CSR

    maxfct = 1; // max number of numerical factorizations
    msglvl = 0; // 1; // No print statistical information
    mnum = 1;   // which factoriszation to use
    error = 0;

    // Initialize the internal solver memory pointer. This is only necessary for the FIRST call.
    // This also stores the factorization of A. So if same  A is to be used for repeated solutions, we can reuse pt.
    for (int i = 0; i < 64; i++)
    {
        pt[i] = 0; // nullptr;
    }
}

Pardiso::~Pardiso()
{
    if (!IsMemoryFreed)
    {
        Terminate();
    }
}

void Pardiso::Factorize(Eigen::SparseMatrix<double, Eigen::RowMajor> &A)
{
    // Reordering and Symbolic Factorization
    phase = Phases::Analysis;
    n = A.rows();
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, A.valuePtr(), A.outerIndexPtr(), A.innerIndexPtr(), perm, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);
    if (error != 0)
    {
        printf("\nERROR during symbolic factorization: %d \n", error);
        exit(1);
    }

    // printf("\nReordering completed ... ");
    // printf("\nNumber of nonzeros in factors = %d", iparm[17]);
    // printf("\nNumber of factorization MFLOPS = %d", iparm[18]);

    // Numerical factorization
    phase = Phases::Factorization;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, A.valuePtr(), A.outerIndexPtr(), A.innerIndexPtr(), perm, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);
    if (error != 0)
    {
        printf("\nERROR during numerical factorization: %d", error);
        // exit(2);
    }
    // printf("\nFactorization completed ... ");

    IsFactorized = true;
}

void Pardiso::SolveSystem(Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    if (IsFactorized)
    {
        // Solve the system
        phase = Phases::Solve_IterativeRefinement;
        pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, &idum, &idum, perm, &nrhs,
                iparm, &msglvl, b.data(), x.data(), &error);

        if (error != 0)
        {
            printf("\nERROR during solution: %d", error);
        }
    }
    else
    {
        std::cerr << "RHS matrix is not factorized yet. Use Factorize()."
                  << "\n";
    }
}

void Pardiso::SolveSystem(Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    Factorize(A);
    SolveSystem(b, x);
    Terminate();
}

void Pardiso::Terminate()
{
    // Terminate the solver
    phase = Phases::ReleaseAllInternalMemory;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, &idum, &idum, perm, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);
    IsMemoryFreed = true;
}
