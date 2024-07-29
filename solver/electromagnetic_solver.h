// The electromagnetic solver solves for the electric potential, the electric
// field, the magnetic vector potential, and the magnetic flux density.

#pragma once

#include <Eigen/Dense>
#include <cstdbool>
#include <cstdlib>
#include <string>

#include "mesh/gmsh_interface.h"
#include "proto/material.pb.h"
#include "proto/solver_config.pb.h"

namespace solver {

// Electromagnetic solver.
template <std::size_t Dimension>
class ElectromagneticSolver : public gmsh::GmshInterface {
 public:
  ElectromagneticSolver(const std::string& mesh_file,
                        solver::SolverConfig solver_config);

  // Get the number of electric potential unknowns.
  std::size_t num_electric_potential_unknowns() const {
    return electric_potential_.size();
  }

  // Get the number of electric field unknowns.
  std::size_t num_electric_field_unknowns() const {
    return electric_field_.size();
  }

  // Get the number of magnetic vector potential unknowns.
  std::size_t num_magnetic_vector_potential_unknowns() const {
    return magnetic_vector_potential_.size();
  }

  // Get the number of magnetic flux density unknowns.
  std::size_t num_magnetic_flux_density_unknowns() const {
    return magnetic_flux_density_.size();
  }

  // Get the number of unknowns.
  std::size_t num_unknowns() const {
    return num_electric_potential_unknowns() +
           num_electric_potential_unknowns() +
           num_magnetic_vector_potential_unknowns() +
           num_magnetic_flux_density_unknowns();
  }

  // Solve for the electric potential, the electric field, the magnetic vector
  // potential, and the magnetic flux density.
  void Solve();

  // Write the solution to a CSV file.
  void WriteSolution(const std::string& csv_file) const;

 protected:
  // Get the material of the given physical group.
  model::Material GetMaterialForPhysicalGroup(int tag);

  // Get the material of the given entity.
  model::Material GetMaterialForEntity(int tag);

  // Validate the mesh.
  virtual void ValidateMesh() {}

  // Implementation for solving the electric potential, the electric field, the
  // magnetic vector potential, and the magnetic flux density.
  virtual void SolveImpl() = 0;

  // Get the index corresponding to the node tag.
  static int node_index_from_tag(const int tag) { return tag - 1; }

  // Get the node tag corresponding to the index.
  static int node_tag_from_index(const int index) { return index + 1; }

  // Solver configuration.
  solver::SolverConfig config_;

  // Number of nodes in the mesh.
  std::size_t num_nodes_ = 0;

  // Electric potential.
  Eigen::VectorXcd electric_potential_;

  // Electric field.
  Eigen::MatrixXcd electric_field_;

  // Magnetic vector potential.
  Eigen::MatrixXcd magnetic_vector_potential_;

  // Magnetic flux density.
  Eigen::MatrixXcd magnetic_flux_density_;

  // If true, the electromagnetic fields have been solved.
  bool solved_ = false;
};

// Type definitions.
using ElectromagneticSolver2D = ElectromagneticSolver<2>;
using ElectromagneticSolver3D = ElectromagneticSolver<3>;

}  // namespace solver
