#include <Eigen/Dense>
#include <Eigen/LU>
#include <cstdlib>

#include "base/logging.h"

int main() {
  base::InitLogging();

  // Solve a matrix-vector equation.
  Eigen::Matrix2f A{{
      {1, 1},
      {1, -1},
  }};
  Eigen::Vector2f b{{5, 3}};
  Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXf> > lu(A);
  Eigen::Vector2f x = lu.solve(b);
  LOG(INFO) << x;

  return EXIT_SUCCESS;
}
