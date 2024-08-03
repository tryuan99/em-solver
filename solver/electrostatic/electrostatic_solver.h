// The electrostatic solver solves for the electric potential and the electric
// field at DC.

#pragma once

#include <cstdlib>
#include <string>

#include "proto/solver_config.pb.h"
#include "solver/electromagnetic_solver.h"

namespace solver {

// Electrostatic solver.
template <std::size_t Dimension>
class ElectrostaticSolver : public ElectromagneticSolver<Dimension> {
 public:
  ElectrostaticSolver(const std::string& mesh_file,
                      const SolverConfig solver_config)
      : ElectromagneticSolver<Dimension>(mesh_file, std::move(solver_config)) {}

  // Get the number of unknowns.
  std::size_t num_unknowns() const override {
    return this->num_electric_potential_unknowns() +
           this->num_electric_field_unknowns();
  }

 protected:
  // Get the DC voltage for the given tag.
  double GetDcVoltage(int tag);
};

// 2D electrostatic solver.
class ElectrostaticSolver2D : public ElectrostaticSolver<2> {
 public:
  ElectrostaticSolver2D(const std::string& mesh_file,
                        const SolverConfig solver_config)
      : ElectrostaticSolver<2>(mesh_file, std::move(solver_config)) {}

 protected:
  // Implementation for solving the electric potential, the electric field, the
  // magnetic vector potential, and the magnetic flux density.
  void SolveImpl() override;

 private:
  // Get the unknown index corresponding to the node's electric potential.
  std::size_t electric_potential_unknown_index(const int tag) {
    return node_index_from_tag(tag);
  }

  // Get the unknown index corresponding to the node's electric field in the
  // x-direction.
  std::size_t electric_field_x_unknown_index(const int tag) {
    return dimension() * node_index_from_tag(tag) +
           num_electric_potential_unknowns();
  }

  // Get the unknown index corresponding to the node's electric field in the
  // y-direction.
  std::size_t electric_field_y_unknown_index(const int tag) {
    return dimension() * node_index_from_tag(tag) + 1 +
           num_electric_potential_unknowns();
  }

  // Get the equation index corresponding to Poisson's equation or any voltage
  // boundary condition.
  std::size_t poisson_electric_potential_equation_index(const int tag) {
    return node_index_from_tag(tag);
  }

  // Get the equation index corresponding to the node's electric field in the
  // x-direction.
  std::size_t electric_field_x_equation_index(const int tag) {
    return dimension() * node_index_from_tag(tag) +
           num_electric_potential_unknowns();
  }

  // Get the equation index corresponding to the node's electric field in the
  // y-direction.
  std::size_t electric_field_y_equation_index(const int tag) {
    return dimension() * node_index_from_tag(tag) + 1 +
           num_electric_potential_unknowns();
  }
};

// 3D electrostatic solver.
class ElectrostaticSolver3D : public ElectrostaticSolver<3> {
 public:
  ElectrostaticSolver3D(const std::string& mesh_file,
                        const SolverConfig solver_config)
      : ElectrostaticSolver<3>(mesh_file, std::move(solver_config)) {}

 protected:
  // Implementation for solving the electric potential, the electric field, the
  // magnetic vector potential, and the magnetic flux density.
  void SolveImpl() override;
};

}  // namespace solver
