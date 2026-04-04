#include "solver/electromagnetic_solver.h"

#include <gmsh.h>

#include <cstdbool>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "absl/strings/str_format.h"
#include "mesh/gmsh_interface.h"
#include "proto/material.pb.h"
#include "proto/solver_config.pb.h"

namespace solver {

template <std::size_t Dimension>
ElectromagneticSolver<Dimension>::ElectromagneticSolver(
    const std::string& mesh_file, const SolverConfig solver_config)
    : config_(std::move(solver_config)) {
  // Open the mesh file.
  gmsh::open(mesh_file);

  // Initialize the electric potential, electric field, magnetic vector
  // potential, and magnetic flux density vectors.
  const auto node_tags = GetNodes(/*tag=*/-1, Dimension);
  node_index_lookup_ = NodeIndexLookup(node_tags);
  num_nodes_ = node_index_lookup_.size();
  electric_potential_ = Eigen::VectorXcd::Zero(num_nodes_);
  electric_field_ = Eigen::MatrixXcd::Zero(num_nodes_, Dimension);
  magnetic_vector_potential_ = Eigen::MatrixXcd::Zero(num_nodes_, Dimension);
  magnetic_flux_density_ =
      Eigen::MatrixXcd::Zero(num_nodes_, Dimension == 3 ? 3 : 1);
  solved_ = false;
}

template <std::size_t Dimension>
void ElectromagneticSolver<Dimension>::Solve() {
  EnsureMeshValidated();
  if (solved_) {
    return;
  }
  SolveImpl();
  solved_ = true;
}

template <std::size_t Dimension>
void ElectromagneticSolver<Dimension>::WriteSolution(
    const std::string& csv_file) const {
  // The columns are in the following order:
  //  - Tag
  //  - Electric potential
  //  - Electric field (x, y, z)
  //  - Magnetic vector potential (x, y, z)
  //  - Magnetic flux density (x, y, z)
  std::ofstream output_csv(csv_file);
  if (!output_csv.is_open()) {
    throw std::runtime_error("Unable to open output CSV file.");
  }
  output_csv << "Tag,Electric potential,"
                "Electric field x,Electric field y,Electric field z,"
                "Magnetic vector potential x,Magnetic vector potential "
                "y,Magnetic vector potential z,"
                "Magnetic flux density x,Magnetic flux density y,Magnetic flux "
                "density z\n";
  const auto complex_number_to_string =
      []<typename T>(const std::complex<T> value) {
        return absl::StrFormat("%g%+gj", value.real(), value.imag());
      };
  for (std::size_t i = 0; i < num_nodes_; ++i) {
    output_csv << node_tag_from_index(i) << ",";
    output_csv << complex_number_to_string(electric_potential_(i)) << ",";
    output_csv << complex_number_to_string(electric_field_(i, 0)) << ","
               << complex_number_to_string(electric_field_(i, 1)) << ","
               << (Dimension > 2
                       ? complex_number_to_string(electric_field_(i, 2))
                       : "0")
               << ",";
    output_csv << complex_number_to_string(magnetic_vector_potential_(i, 0))
               << ","
               << complex_number_to_string(magnetic_vector_potential_(i, 1))
               << ","
               << (Dimension > 2 ? complex_number_to_string(
                                       magnetic_vector_potential_(i, 2))
                                 : "0")
               << ",";
    output_csv << (Dimension > 2
                       ? complex_number_to_string(magnetic_flux_density_(i, 0))
                       : "0")
               << ","
               << (Dimension > 2
                       ? complex_number_to_string(magnetic_flux_density_(i, 1))
                       : "0")
               << ","
               << (Dimension > 2
                       ? complex_number_to_string(magnetic_flux_density_(i, 2))
                       : complex_number_to_string(
                             magnetic_flux_density_(i, 0)));
    output_csv << "\n";
  }
}

template <std::size_t Dimension>
model::Material ElectromagneticSolver<Dimension>::GetMaterialForPhysicalGroup(
    const int tag) const {
  std::string physical_group_name;
  gmsh::model::getPhysicalName(Dimension, tag, physical_group_name);
  model::Material material = model::Material::UNSPECIFIED;
  if (!model::Material_Parse(physical_group_name, &material)) {
    throw std::invalid_argument("Invalid material.");
  }
  return material;
}

template <std::size_t Dimension>
model::Material ElectromagneticSolver<Dimension>::GetMaterialForEntity(
    const int tag) const {
  const auto physical_group_tags = GetPhysicalGroupsForEntity(Dimension, tag);
  if (physical_group_tags.empty()) {
    throw std::invalid_argument(
        absl::StrFormat("Entity %d cannot be found.", tag));
  }
  if (physical_group_tags.size() > 1) {
    throw std::invalid_argument(
        absl::StrFormat("Entity %d belongs to multiple physical groups.", tag));
  }
  const auto physical_group_tag = physical_group_tags[0];
  return GetMaterialForPhysicalGroup(physical_group_tag);
}

// Explicit instantiations of the template.
template class ElectromagneticSolver<2>;
template class ElectromagneticSolver<3>;

}  // namespace solver
