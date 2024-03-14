#include "Pardiso.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

// using myMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, MKL_INT>;

using myMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

void set_a(myMatrix &mat)
{
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> tripletList;
    // tripletList.reserve(nz);
    tripletList.push_back(Triplet(0, 0, 1));
    tripletList.push_back(Triplet(0, 1, -1));
    tripletList.push_back(Triplet(0, 3, -3));
    tripletList.push_back(Triplet(1, 0, -2));
    tripletList.push_back(Triplet(1, 1, 5));
    tripletList.push_back(Triplet(2, 2, 4));
    tripletList.push_back(Triplet(2, 3, 6));
    tripletList.push_back(Triplet(2, 4, 4));
    tripletList.push_back(Triplet(3, 0, -3));
    tripletList.push_back(Triplet(3, 2, 6));
    tripletList.push_back(Triplet(3, 3, 7));
    tripletList.push_back(Triplet(4, 1, 8));
    tripletList.push_back(Triplet(4, 4, -5));
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    mat.makeCompressed();
}

void set_b(Eigen::VectorXd &b)
{
    int n = b.size();
    for (int i = 0; i < n; i++)
        b[i] = 1;
}

int main()
{

    /**-------------Example 1 ------------------*/
    // Create matrices. Note, it must be Eigen::RowMajor.
    int size = 5;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(5, 5);

    A.coeffRef(0, 0) = 2.0;
    A.coeffRef(0, 2) = 3;
    A.coeffRef(0, 4) = 9.0;
    A.coeffRef(1, 0) = 5.;
    A.coeffRef(1, 1) = 9.0;
    A.coeffRef(1, 2) = 7.0;
    A.coeffRef(2, 0) = 12.0;
    A.coeffRef(2, 1) = 14.0;
    A.coeffRef(2, 2) = 68.0;
    A.coeffRef(3, 1) = 12.0;
    A.coeffRef(3, 3) = 14.0;
    A.coeffRef(3, 4) = 68.0;
    A.coeffRef(4, 0) = 12.0;
    A.coeffRef(4, 2) = 14.0;
    A.coeffRef(4, 4) = 68.0;
    A.makeCompressed();

    Eigen::VectorXd b(size), x(size);
    b << 1, 2, 3, 4, 5;

    Pardiso solver(Pardiso::MatrixTypes::UnsymmetricReal);
    solver.SolveSystem(A, b, x);

    std::cout << "A:: \n"
              << A << "\n";
    std::cout << "b:: \n"
              << b << "\n";
    std::cout << "x: \n"
              << x << "\n";

    /**-------------Example 2 ------------------*/
    // set data
    int n = 5;
    myMatrix P(n, n);
    set_a(P);
    Eigen::VectorXd q(n), r(n);
    set_b(q);

    // reuse the same solver.
    solver.SolveSystem(P, q, r);

    std::cout << "P:: \n"
              << P << "\n";
    std::cout << "q:: \n"
              << q << "\n";
    std::cout << "r: \n"
              << r << "\n";

    /**-------------Example 3 ------------------*/

    // Here we will factorize P. Then we can simply do step 2 with different RHS, and then finally end
    Eigen::VectorXd solution(n);

    solver.Factorize(P);
    solver.SolveSystem(b, solution); // # for first rhs

    std::cout << "For first RHS \n";
    std::cout << "P:: \n"
              << P << "\n";
    std::cout << "m:: \n"
              << b << "\n";
    std::cout << "n: \n"
              << solution << "\n";

    std::cout << "For second RHS using the factorization already performed. \n";
    solver.SolveSystem(q, solution);
    std::cout << "P:: \n"
              << P << "\n";
    std::cout << "q:: \n"
              << q << "\n";
    std::cout << "n: \n"
              << solution << "\n";
    solver.Terminate(); // use this to finally free since we wont use the factorization anymore

    return 0;
}