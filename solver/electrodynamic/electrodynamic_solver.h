// The electrodynamic solver solves for the electric potential, the electric
// field, the magnetic vector potential, and the magnetic flux density at a
// given frequency.

#pragma once

#include <complex>
#include <cstdlib>
#include <string>

#include "proto/solver_config.pb.h"
#include "solver/electromagnetic_solver.h"

namespace solver {

// Electrodynamic solver.
template <std::size_t Dimension>
class ElectrodynamicSolver : public ElectromagneticSolver<Dimension> {
 public:
  ElectrodynamicSolver(const std::string& mesh_file,
                       const SolverConfig solver_config)
      : ElectromagneticSolver<Dimension>(mesh_file, std::move(solver_config)) {}

  // Get the number of unknowns.
  std::size_t num_unknowns() const override {
    return this->num_electric_potential_unknowns() +
           this->num_electric_field_unknowns() +
           this->num_magnetic_vector_potential_unknowns() +
           this->num_magnetic_flux_density_unknowns() + this->num_nodes_;
  }

 protected:
  // Get the AC phasor voltage for the tag.
  std::complex<double> GetAcPhasor(int tag) const;
};

// 2D electrodynamic solver.
class ElectrodynamicSolver2D : public ElectrodynamicSolver<2> {
 public:
  ElectrodynamicSolver2D(const std::string& mesh_file,
                         const SolverConfig solver_config)
      : ElectrodynamicSolver<2>(mesh_file, std::move(solver_config)) {}

 protected:
  // Implementation for solving the electric potential, the electric field, the
  // magnetic vector potential, and the magnetic flux density.
  void SolveImpl() override;

 private:
  // Get the unknown index corresponding to the node's electric potential.
  std::size_t electric_potential_unknown_index(const int tag) {
    return this->node_index_from_tag(tag);
  }

  // Get the unknown index corresponding to the node's electric field in the
  // x-direction.
  std::size_t electric_field_x_unknown_index(const int tag) {
    return dimension() * this->node_index_from_tag(tag) +
           num_electric_potential_unknowns();
  }

  // Get the unknown index corresponding to the node's electric field in the
  // y-direction.
  std::size_t electric_field_y_unknown_index(const int tag) {
    return dimension() * this->node_index_from_tag(tag) + 1 +
           num_electric_potential_unknowns();
  }

  // Get the unknown index corresponding to the node's magnetic vector potential
  // in the x-direction.
  std::size_t magnetic_vector_potential_x_unknown_index(const int tag) {
    return dimension() * this->node_index_from_tag(tag) +
           num_electric_potential_unknowns() + num_electric_field_unknowns();
  }

  // Get the unknown index corresponding to the node's magnetic vector potential
  // in the y-direction.
  std::size_t magnetic_vector_potential_y_unknown_index(const int tag) {
    return dimension() * this->node_index_from_tag(tag) + 1 +
           num_electric_potential_unknowns() + num_electric_field_unknowns();
  }

  // Get the unknown index corresponding to the node's magnetic flux density in
  // the z-direction.
  std::size_t magnetic_flux_density_z_unknown_index(const int tag) {
    return this->node_index_from_tag(tag) + num_electric_potential_unknowns() +
           num_electric_field_unknowns() +
           num_magnetic_vector_potential_unknowns();
  }

  // Get the unknown index corresponding to the gauge.
  std::size_t gauge_unknown_index(const int tag) {
    return this->node_index_from_tag(tag) + num_electric_potential_unknowns() +
           num_electric_field_unknowns() +
           num_magnetic_vector_potential_unknowns() +
           num_magnetic_flux_density_unknowns();
  }

  // Get the equation index corresponding to Gauss's law or any voltage boundary
  // condition.
  std::size_t gauss_law_electric_potential_equation_index(const int tag) {
    return this->node_index_from_tag(tag);
  }

  // Get the equation index corresponding to Faraday's law in the x-direction.
  std::size_t faraday_law_x_equation_index(const int tag) {
    return dimension() * this->node_index_from_tag(tag) +
           num_electric_potential_unknowns();
  }

  // Get the equation index corresponding to Faraday's law in the y-direction.
  std::size_t faraday_law_y_equation_index(const int tag) {
    return dimension() * this->node_index_from_tag(tag) + 1 +
           num_electric_potential_unknowns();
  }

  // Get the equation index corresponding to Gauss's law for magnetism in the
  // z-direction.
  std::size_t gauss_law_magnetism_z_equation_index(const int tag) {
    return this->node_index_from_tag(tag) + num_electric_potential_unknowns() +
           num_electric_field_unknowns();
  }

  // Get the equation index corresponding to Ampere's law in the x-direction.
  std::size_t ampere_law_x_equation_index(const int tag) {
    return dimension() * this->node_index_from_tag(tag) +
           num_electric_potential_unknowns() + num_electric_field_unknowns() +
           num_magnetic_flux_density_unknowns();
  }

  // Get the equation index corresponding to Ampere's law in the y-direction.
  std::size_t ampere_law_y_equation_index(const int tag) {
    return dimension() * this->node_index_from_tag(tag) + 1 +
           num_electric_potential_unknowns() + num_electric_field_unknowns() +
           num_magnetic_flux_density_unknowns();
  }

  // Get the equation index corresponding to the gauge.
  std::size_t gauge_equation_index(const int tag) {
    return this->node_index_from_tag(tag) + num_electric_potential_unknowns() +
           num_electric_field_unknowns() +
           num_magnetic_flux_density_unknowns() +
           num_magnetic_vector_potential_unknowns();
  }
};

// 3D electrodynamic solver.
class ElectrodynamicSolver3D : public ElectrodynamicSolver<3> {
 public:
  ElectrodynamicSolver3D(const std::string& mesh_file,
                         const SolverConfig solver_config)
      : ElectrodynamicSolver<3>(mesh_file, std::move(solver_config)) {}

 protected:
  // Implementation for solving the electric potential, the electric field, the
  // magnetic vector potential, and the magnetic flux density.
  void SolveImpl() override;
};

}  // namespace solver
