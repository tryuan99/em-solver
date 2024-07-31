#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <complex>
#include <cstdlib>
#include <forward_list>

#include "absl/strings/str_format.h"
#include "base/base.h"

int main(int argc, char** argv) {
  base::Init(argc, argv);

  // Set up a sparse matrix-vector equation.
  std::forward_list<Eigen::Triplet<std::complex<double>>> A_triplets{
      {0, 0, 1}, {0, 1, -1}, {1, 1, 1}};
  Eigen::SparseMatrix<std::complex<double>> A(2, 2);
  A.setFromTriplets(A_triplets.cbegin(), A_triplets.cend());
  std::forward_list<Eigen::Triplet<std::complex<double>>> b_triplets{{0, 0, -7},
                                                                     {1, 0, 3}};
  Eigen::SparseMatrix<std::complex<double>> b(2, 1);
  b.setFromTriplets(b_triplets.cbegin(), b_triplets.cend());

  // Solve the sparse matrix-vector equation.
  Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(absl::StrFormat(
        "Failed to compute the decomposition of the matrix: %d.",
        solver.info()));
  }
  Eigen::VectorXcd x = solver.solve(b);
  LOG(INFO) << x;

  return EXIT_SUCCESS;
}
