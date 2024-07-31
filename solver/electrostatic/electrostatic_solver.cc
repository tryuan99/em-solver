#include "solver/electrostatic/electrostatic_solver.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <complex>
#include <forward_list>
#include <stdexcept>
#include <unordered_set>

#include "absl/strings/str_format.h"
#include "mesh/gmsh_interface.h"
#include "model/material.h"
#include "solver/neighbor_lookup.h"

namespace solver {

template <std::size_t Dimension>
double ElectrostaticSolver<Dimension>::GetDcVoltage(const int tag) {
  for (const auto& entity_config : this->config_.entity_configs()) {
    if (entity_config.tag() == tag) {
      return entity_config.dc_voltage();
    }
  }
  throw std::invalid_argument(
      absl::StrFormat("Entity %d cannot be found.", tag));
}

void ElectrostaticSolver2D::SolveImpl() {
  // In the matrix-vector equation, the first unknowns correspond to the nodes'
  // electric potentials, and the remaining unknowns correspond to the nodes'
  // electric fields.
  // Similarly, the first equations correspond to Poisson's equation and any
  // voltage boundary conditions, and the remaining equations correspond to the
  // electric field equations for each node in the x and y-directions.

  // Get the node coordinates.
  const auto node_tag_to_coordinates =
      GetNodeCoordinates(/*tag=*/-1, dimension());

  // Get the conductor and the insulator entities.
  const auto physical_groups = GetPhysicalGroups(dimension());
  std::unordered_set<int> conductor_entity_tags;
  std::unordered_set<int> insulator_entity_tags;
  for (const auto& [physical_group_dimension, physical_group_tag] :
       physical_groups) {
    const auto material = GetMaterialForPhysicalGroup(physical_group_tag);
    const auto& material_properties =
        model::MaterialProperties::kMaterialToProperties.at(material);
    const auto entities =
        GetEntitiesForPhysicalGroups(dimension(), physical_group_tag);
    if (material_properties.is_conductor()) {
      conductor_entity_tags.insert(entities.cbegin(), entities.cend());
    } else if (material_properties.is_insulator()) {
      insulator_entity_tags.insert(entities.cbegin(), entities.cend());
    }
  }

  // Initialize the matrix-vector triplets.
  std::forward_list<Eigen::Triplet<std::complex<double>>> A_triplets;
  std::forward_list<Eigen::Triplet<std::complex<double>>> b_triplets;

  // Fill in the electric potential and electric field equations for the nodes
  // within the conductors. For the boundary nodes, only fill in the electric
  // potential equations.
  // At steady state, the electric potential is constant throughout the
  // insulator, and the electric field is zero throughout, even with a non-zero
  // resistivity.
  std::unordered_set<gmsh::Tag> boundary_tags;
  for (const auto conductor_tag : conductor_entity_tags) {
    // Set the voltage boundary conditions and electric fields within the
    // conductor.
    const auto conductor_internal_node_tags =
        GetNodes(conductor_tag, dimension(), gmsh::NodeType::kInternal);
    for (const auto tag : conductor_internal_node_tags) {
      // Equation indices.
      const auto poisson_electric_potential_equation_index =
          this->poisson_electric_potential_equation_index(tag);
      const auto electric_field_x_equation_index =
          this->electric_field_x_equation_index(tag);
      const auto electric_field_y_equation_index =
          this->electric_field_y_equation_index(tag);

      // Unknown indices.
      const auto electric_potential_unknown_index =
          this->electric_potential_unknown_index(tag);
      const auto electric_field_x_unknown_index =
          this->electric_field_x_unknown_index(tag);
      const auto electric_field_y_unknown_index =
          this->electric_field_y_unknown_index(tag);

      // Set the voltage boundary conditions.
      A_triplets.emplace_front(poisson_electric_potential_equation_index,
                               electric_potential_unknown_index, 1);
      b_triplets.emplace_front(poisson_electric_potential_equation_index, 0,
                               GetDcVoltage(conductor_tag));

      // Set the electric field boundary conditions.
      A_triplets.emplace_front(electric_field_x_equation_index,
                               electric_field_x_unknown_index, 1);
      A_triplets.emplace_front(electric_field_y_equation_index,
                               electric_field_y_unknown_index, 1);
    }

    // Set the voltage boundary conditions for the boundary nodes of the
    // conductor.
    const auto conductor_boundary_node_tags =
        GetNodes(conductor_tag, dimension(), gmsh::NodeType::kBoundary);
    for (const auto tag : conductor_boundary_node_tags) {
      // Equation indices.
      const auto poisson_electric_potential_equation_index =
          this->poisson_electric_potential_equation_index(tag);

      // Unknown indices.
      const auto electric_potential_unknown_index =
          this->electric_potential_unknown_index(tag);

      // Set the voltage boundary conditions.
      A_triplets.emplace_front(poisson_electric_potential_equation_index,
                               electric_potential_unknown_index, 1);
      b_triplets.emplace_front(poisson_electric_potential_equation_index, 0,
                               GetDcVoltage(conductor_tag));
    }
    boundary_tags.insert(conductor_boundary_node_tags.cbegin(),
                         conductor_boundary_node_tags.cend());
  }

  // Fill in Poisson's equations and the electric field equations for the nodes
  // within the insulators, including the boundary nodes.
  for (const auto insulator_tag : insulator_entity_tags) {
    const auto insulator_triangles = GetFaces(insulator_tag);
    const auto insulator_triangle_neighbors =
        NeighborLookup2D(insulator_triangles);
    const auto insulator_node_tags =
        GetNodes(insulator_tag, dimension(), gmsh::NodeType::kAll);
    for (const auto tag : insulator_node_tags) {
      const auto& node_coordinates = node_tag_to_coordinates.at(tag);
      const auto x = node_coordinates[0];
      const auto y = node_coordinates[1];
      const auto num_adjacent_triangles =
          insulator_triangle_neighbors.num_adjacent_entities(tag);

      // Equation indices.
      const auto poisson_electric_potential_equation_index =
          this->poisson_electric_potential_equation_index(tag);
      const auto electric_field_x_equation_index =
          this->electric_field_x_equation_index(tag);
      const auto electric_field_y_equation_index =
          this->electric_field_y_equation_index(tag);

      // Unknown indices.
      const auto electric_potential_unknown_index =
          this->electric_potential_unknown_index(tag);
      const auto electric_field_x_unknown_index =
          this->electric_field_x_unknown_index(tag);
      const auto electric_field_y_unknown_index =
          this->electric_field_y_unknown_index(tag);

      // Iterate over all adjacent triangles.
      for (const auto& [tag_neighbor1, tag_neighbor2] :
           insulator_triangle_neighbors.neighbors(tag)) {
        const auto& node_coordinates_neighbor1 =
            node_tag_to_coordinates.at(tag_neighbor1);
        const auto x_neighbor1 = node_coordinates_neighbor1[0];
        const auto y_neighbor1 = node_coordinates_neighbor1[1];
        const auto& node_coordinates_neighbor2 =
            node_tag_to_coordinates.at(tag_neighbor2);
        const auto x_neighbor2 = node_coordinates_neighbor2[0];
        const auto y_neighbor2 = node_coordinates_neighbor2[1];

        // Unknown indices for both neighbors.
        const auto electric_potential_unknown_index_neighbor1 =
            this->electric_potential_unknown_index(tag_neighbor1);
        const auto electric_potential_unknown_index_neighbor2 =
            this->electric_potential_unknown_index(tag_neighbor2);
        const auto electric_field_x_unknown_index_neighbor1 =
            this->electric_field_x_unknown_index(tag_neighbor1);
        const auto electric_field_x_unknown_index_neighbor2 =
            this->electric_field_x_unknown_index(tag_neighbor2);
        const auto electric_field_y_unknown_index_neighbor1 =
            this->electric_field_y_unknown_index(tag_neighbor1);
        const auto electric_field_y_unknown_index_neighbor2 =
            this->electric_field_y_unknown_index(tag_neighbor2);

        // Calculate the denominator.
        const auto denominator = (x_neighbor1 * y_neighbor2 - x_neighbor1 * y -
                                  x_neighbor2 * y_neighbor1 + x_neighbor2 * y +
                                  x * y_neighbor1 - x * y_neighbor2);

        // Add the coefficients of the electric fields for Poisson's equation.
        // Boundary nodes already have a voltage boundary condition, so skip
        // boundary nodes.
        if (!boundary_tags.contains(tag)) {
          A_triplets.emplace_front(poisson_electric_potential_equation_index,
                                   electric_field_x_unknown_index_neighbor1,
                                   (y_neighbor2 - y) / denominator);
          A_triplets.emplace_front(poisson_electric_potential_equation_index,
                                   electric_field_y_unknown_index_neighbor1,
                                   (x - x_neighbor2) / denominator);
          A_triplets.emplace_front(poisson_electric_potential_equation_index,
                                   electric_field_x_unknown_index_neighbor2,
                                   (y - y_neighbor1) / denominator);
          A_triplets.emplace_front(poisson_electric_potential_equation_index,
                                   electric_field_y_unknown_index_neighbor2,
                                   (x_neighbor1 - x) / denominator);
          A_triplets.emplace_front(poisson_electric_potential_equation_index,
                                   electric_field_x_unknown_index,
                                   (y_neighbor1 - y_neighbor2) / denominator);
          A_triplets.emplace_front(poisson_electric_potential_equation_index,
                                   electric_field_y_unknown_index,
                                   (x_neighbor2 - x_neighbor1) / denominator);
        }

        // Add the coefficients of the electric potentials for the electric
        // field equation in the x-direction.
        A_triplets.emplace_front(electric_field_x_equation_index,
                                 electric_potential_unknown_index_neighbor1,
                                 (y_neighbor2 - y) / denominator);
        A_triplets.emplace_front(electric_field_x_equation_index,
                                 electric_potential_unknown_index_neighbor2,
                                 (y - y_neighbor1) / denominator);
        A_triplets.emplace_front(electric_field_x_equation_index,
                                 electric_potential_unknown_index,
                                 (y_neighbor1 - y_neighbor2) / denominator);

        // Add the coefficients of the electric potentials for the electric
        // field equation in the y-direction.
        A_triplets.emplace_front(electric_field_y_equation_index,
                                 electric_potential_unknown_index_neighbor1,
                                 (x - x_neighbor2) / denominator);
        A_triplets.emplace_front(electric_field_y_equation_index,
                                 electric_potential_unknown_index_neighbor2,
                                 (x_neighbor1 - x) / denominator);
        A_triplets.emplace_front(electric_field_y_equation_index,
                                 electric_potential_unknown_index,
                                 (x_neighbor2 - x_neighbor1) / denominator);
      }

      // Set the coefficient for the electric fields for the electric field
      // equations.
      A_triplets.emplace_front(electric_field_x_equation_index,
                               electric_field_x_unknown_index,
                               num_adjacent_triangles);
      A_triplets.emplace_front(electric_field_y_equation_index,
                               electric_field_y_unknown_index,
                               num_adjacent_triangles);
    }
  }

  // Create the sparse matrix-vector equation.
  Eigen::SparseMatrix<std::complex<double>> A(num_unknowns(), num_unknowns());
  A.setFromTriplets(A_triplets.cbegin(), A_triplets.cend());
  Eigen::SparseMatrix<std::complex<double>> b(num_unknowns(), 1);
  b.setFromTriplets(b_triplets.cbegin(), b_triplets.cend());

  // Solve for the electric potential and the electric field.
  Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(absl::StrFormat(
        "Failed to compute the decomposition of the matrix: %d.",
        solver.info()));
  }
  Eigen::VectorXcd x = solver.solve(b);
  electric_potential_ = x(Eigen::seqN(0, num_electric_potential_unknowns()));
  electric_field_ = x(Eigen::seqN(num_electric_potential_unknowns(),
                                  num_electric_field_unknowns()))
                        .reshaped<Eigen::RowMajor>(num_nodes_, dimension());
}

}  // namespace solver
