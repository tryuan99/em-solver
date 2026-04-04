#include "solver/electrodynamic/electrodynamic_solver.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SuperLUSupport>
#include <complex>
#include <cstdlib>
#include <forward_list>
#include <numbers>
#include <stdexcept>
#include <unordered_set>

#include "absl/strings/str_format.h"
#include "mesh/gmsh_interface.h"
#include "model/material.h"
#include "solver/neighbor_lookup.h"

namespace solver {

template <std::size_t Dimension>
std::complex<double> ElectrodynamicSolver<Dimension>::GetAcPhasor(
    const int tag) const {
  for (const auto& entity_config : this->config_.entity_configs()) {
    if (entity_config.tag() == tag) {
      return std::complex<double>(entity_config.ac_phasor().real(),
                                  entity_config.ac_phasor().imaginary());
    }
  }
  throw std::invalid_argument(
      absl::StrFormat("Entity %d cannot be found.", tag));
}

void ElectrodynamicSolver2D::SolveImpl() {
  // In the matrix-vector equation, the first unknowns correspond to the nodes'
  // electric potentials, the next unknowns correspond to the nodes' electric
  // fields, and the next unknowns correspond to the nodes' magnetic vector
  // potentials, the next unknowns correspond to the nodes' magnetic flux
  // densities, and the last unknowns correspond to the Lagrange multiplier for
  // the gauge.
  // Similarly, the first equations correspond to Gauss's law and voltage
  // boundary conditions, the next equations correspond to Faraday's law, the
  // next equations correspond to Gauss's law for magnetism, the next equations
  // correspond to Ampere's law, and the last equation corresponds to the
  // Lagrange multiplier for the gauge.
  const auto omega = 2 * std::numbers::pi * config_.frequency();

  // Get the node coordinates.
  const auto node_tag_to_coordinates =
      GetNodeCoordinates(/*tag=*/-1, dimension());

  // Get the conductor and the insulator entities.
  const auto physical_groups = GetPhysicalGroups(dimension());
  std::unordered_set<int> conductor_entity_tags;
  std::unordered_set<int> insulator_entity_tags;
  for (const auto& physical_group : physical_groups) {
    const auto physical_group_tag = physical_group.second;
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

  // Fill in the equations for all nodes.
  const auto triangles = GetFaces(/*tag=*/-1);
  const auto triangle_neighbors = NeighborLookup2D(triangles);
  for (const auto& [tag, node_coordinates] : node_tag_to_coordinates) {
    const auto x = node_coordinates[0];
    const auto y = node_coordinates[1];
    const auto num_adjacent_triangles =
        triangle_neighbors.num_adjacent_entities(tag);

    // Equation indices.
    const auto gauss_law_magnetism_z_equation_index =
        this->gauss_law_magnetism_z_equation_index(tag);

    // Unknown indices.
    const auto magnetic_vector_potential_x_unknown_index =
        this->magnetic_vector_potential_x_unknown_index(tag);
    const auto magnetic_vector_potential_y_unknown_index =
        this->magnetic_vector_potential_y_unknown_index(tag);
    const auto magnetic_flux_density_z_unknown_index =
        this->magnetic_flux_density_z_unknown_index(tag);

    // Iterate over all adjacent triangles.
    for (const auto& [tag_neighbor1, tag_neighbor2] :
         triangle_neighbors.neighbors(tag)) {
      const auto& node_coordinates_neighbor1 =
          node_tag_to_coordinates.at(tag_neighbor1);
      const auto x_neighbor1 = node_coordinates_neighbor1[0];
      const auto y_neighbor1 = node_coordinates_neighbor1[1];
      const auto& node_coordinates_neighbor2 =
          node_tag_to_coordinates.at(tag_neighbor2);
      const auto x_neighbor2 = node_coordinates_neighbor2[0];
      const auto y_neighbor2 = node_coordinates_neighbor2[1];

      // Unknown indices for both neighbors.
      const auto magnetic_vector_potential_x_unknown_index_neighbor1 =
          this->magnetic_vector_potential_x_unknown_index(tag_neighbor1);
      const auto magnetic_vector_potential_x_unknown_index_neighbor2 =
          this->magnetic_vector_potential_x_unknown_index(tag_neighbor2);
      const auto magnetic_vector_potential_y_unknown_index_neighbor1 =
          this->magnetic_vector_potential_y_unknown_index(tag_neighbor1);
      const auto magnetic_vector_potential_y_unknown_index_neighbor2 =
          this->magnetic_vector_potential_y_unknown_index(tag_neighbor2);

      // Calculate the denominator.
      const auto denominator = (x_neighbor1 * y_neighbor2 - x_neighbor1 * y -
                                x_neighbor2 * y_neighbor1 + x_neighbor2 * y +
                                x * y_neighbor1 - x * y_neighbor2);

      // Add the coefficients for the equation corresponding to Gauss's law for
      // magnetism in the z-direction.
      A_triplets.emplace_front(
          gauss_law_magnetism_z_equation_index,
          magnetic_vector_potential_x_unknown_index_neighbor1,
          (x - x_neighbor2) / denominator);
      A_triplets.emplace_front(
          gauss_law_magnetism_z_equation_index,
          magnetic_vector_potential_y_unknown_index_neighbor1,
          -(y_neighbor2 - y) / denominator);
      A_triplets.emplace_front(
          gauss_law_magnetism_z_equation_index,
          magnetic_vector_potential_x_unknown_index_neighbor2,
          (x_neighbor1 - x) / denominator);
      A_triplets.emplace_front(
          gauss_law_magnetism_z_equation_index,
          magnetic_vector_potential_y_unknown_index_neighbor2,
          -(y - y_neighbor1) / denominator);
      A_triplets.emplace_front(gauss_law_magnetism_z_equation_index,
                               magnetic_vector_potential_x_unknown_index,
                               (x_neighbor2 - x_neighbor1) / denominator);
      A_triplets.emplace_front(gauss_law_magnetism_z_equation_index,
                               magnetic_vector_potential_y_unknown_index,
                               -(y_neighbor1 - y_neighbor2) / denominator);
    }

    // Set the coefficients for the magnetic flux densities for the equation
    // corresponding to Gauss's law for magnetism.
    A_triplets.emplace_front(gauss_law_magnetism_z_equation_index,
                             magnetic_flux_density_z_unknown_index,
                             num_adjacent_triangles);
  }

  // Fill in the equations for the nodes within the conductors. For the boundary
  // nodes, only fill in the voltage boundary conditions and Ampere's law.
  std::unordered_set<gmsh::Tag> boundary_tags;
  for (const auto conductor_tag : conductor_entity_tags) {
    const auto conductor_properties =
        model::MaterialProperties::kMaterialToProperties.at(
            GetMaterialForEntity(conductor_tag));
    const auto conductor_triangles = GetFaces(conductor_tag);
    const auto conductor_triangle_neighbors =
        NeighborLookup2D(conductor_triangles);

    // Fill in the equations for the nodes within the conductors.
    const auto conductor_internal_node_tags =
        GetNodes(conductor_tag, dimension(), gmsh::NodeType::kInternal);
    for (const auto tag : conductor_internal_node_tags) {
      const auto& node_coordinates = node_tag_to_coordinates.at(tag);
      const auto x = node_coordinates[0];
      const auto y = node_coordinates[1];
      const auto num_adjacent_triangles =
          conductor_triangle_neighbors.num_adjacent_entities(tag);

      // Equation indices.
      const auto gauss_law_electric_potential_equation_index =
          this->gauss_law_electric_potential_equation_index(tag);
      const auto faraday_law_x_equation_index =
          this->faraday_law_x_equation_index(tag);
      const auto faraday_law_y_equation_index =
          this->faraday_law_y_equation_index(tag);
      const auto ampere_law_x_equation_index =
          this->ampere_law_x_equation_index(tag);
      const auto ampere_law_y_equation_index =
          this->ampere_law_y_equation_index(tag);
      const auto gauge_equation_index = this->gauge_equation_index(tag);

      // Unknown indices.
      const auto electric_potential_unknown_index =
          this->electric_potential_unknown_index(tag);
      const auto electric_field_x_unknown_index =
          this->electric_field_x_unknown_index(tag);
      const auto electric_field_y_unknown_index =
          this->electric_field_y_unknown_index(tag);
      const auto magnetic_vector_potential_x_unknown_index =
          this->magnetic_vector_potential_x_unknown_index(tag);
      const auto magnetic_vector_potential_y_unknown_index =
          this->magnetic_vector_potential_y_unknown_index(tag);
      const auto magnetic_flux_density_z_unknown_index =
          this->magnetic_flux_density_z_unknown_index(tag);
      const auto gauge_unknown_index = this->gauge_unknown_index(tag);

      // Set the voltage boundary conditions.
      A_triplets.emplace_front(gauss_law_electric_potential_equation_index,
                               electric_potential_unknown_index, 1);
      b_triplets.emplace_front(gauss_law_electric_potential_equation_index, 0,
                               GetAcPhasor(conductor_tag));

      // Iterate over all adjacent triangles.
      for (const auto& [tag_neighbor1, tag_neighbor2] :
           conductor_triangle_neighbors.neighbors(tag)) {
        const auto& node_coordinates_neighbor1 =
            node_tag_to_coordinates.at(tag_neighbor1);
        const auto x_neighbor1 = node_coordinates_neighbor1[0];
        const auto y_neighbor1 = node_coordinates_neighbor1[1];
        const auto& node_coordinates_neighbor2 =
            node_tag_to_coordinates.at(tag_neighbor2);
        const auto x_neighbor2 = node_coordinates_neighbor2[0];
        const auto y_neighbor2 = node_coordinates_neighbor2[1];

        // Unknown indices for both neighbors.
        const auto magnetic_vector_potential_x_unknown_index_neighbor1 =
            this->magnetic_vector_potential_x_unknown_index(tag_neighbor1);
        const auto magnetic_vector_potential_x_unknown_index_neighbor2 =
            this->magnetic_vector_potential_x_unknown_index(tag_neighbor2);
        const auto magnetic_vector_potential_y_unknown_index_neighbor1 =
            this->magnetic_vector_potential_y_unknown_index(tag_neighbor1);
        const auto magnetic_vector_potential_y_unknown_index_neighbor2 =
            this->magnetic_vector_potential_y_unknown_index(tag_neighbor2);
        const auto magnetic_flux_density_z_unknown_index_neighbor1 =
            this->magnetic_flux_density_z_unknown_index(tag_neighbor1);
        const auto magnetic_flux_density_z_unknown_index_neighbor2 =
            this->magnetic_flux_density_z_unknown_index(tag_neighbor2);
        const auto gauge_unknown_index_neighbor1 =
            this->gauge_unknown_index(tag_neighbor1);
        const auto gauge_unknown_index_neighbor2 =
            this->gauge_unknown_index(tag_neighbor2);

        // Calculate the denominator.
        const auto denominator = (x_neighbor1 * y_neighbor2 - x_neighbor1 * y -
                                  x_neighbor2 * y_neighbor1 + x_neighbor2 * y +
                                  x * y_neighbor1 - x * y_neighbor2);

        // Add the coefficients for the equation corresponding to Ampere's law
        // in the x-direction.
        A_triplets.emplace_front(
            ampere_law_x_equation_index,
            magnetic_flux_density_z_unknown_index_neighbor1,
            -(x - x_neighbor2) / denominator);
        A_triplets.emplace_front(
            ampere_law_x_equation_index,
            magnetic_flux_density_z_unknown_index_neighbor2,
            -(x_neighbor1 - x) / denominator);
        A_triplets.emplace_front(ampere_law_x_equation_index,
                                 magnetic_flux_density_z_unknown_index,
                                 -(x_neighbor2 - x_neighbor1) / denominator);
        A_triplets.emplace_front(ampere_law_x_equation_index,
                                 gauge_unknown_index_neighbor1,
                                 (y_neighbor2 - y) / denominator);
        A_triplets.emplace_front(ampere_law_x_equation_index,
                                 gauge_unknown_index_neighbor2,
                                 (y - y_neighbor1) / denominator);
        A_triplets.emplace_front(ampere_law_x_equation_index,
                                 gauge_unknown_index,
                                 (y_neighbor1 - y_neighbor2) / denominator);

        // Add the coefficients for the equation corresponding to Ampere's law
        // in the y-direction.
        A_triplets.emplace_front(
            ampere_law_y_equation_index,
            magnetic_flux_density_z_unknown_index_neighbor1,
            (y_neighbor2 - y) / denominator);
        A_triplets.emplace_front(
            ampere_law_y_equation_index,
            magnetic_flux_density_z_unknown_index_neighbor2,
            (y - y_neighbor1) / denominator);
        A_triplets.emplace_front(ampere_law_y_equation_index,
                                 magnetic_flux_density_z_unknown_index,
                                 (y_neighbor1 - y_neighbor2) / denominator);
        A_triplets.emplace_front(ampere_law_y_equation_index,
                                 gauge_unknown_index_neighbor1,
                                 (x - x_neighbor2) / denominator);
        A_triplets.emplace_front(ampere_law_y_equation_index,
                                 gauge_unknown_index_neighbor2,
                                 (x_neighbor1 - x) / denominator);
        A_triplets.emplace_front(ampere_law_y_equation_index,
                                 gauge_unknown_index,
                                 (x_neighbor2 - x_neighbor1) / denominator);

        // Add the coefficients for the equation corresponding to the gauge.
        A_triplets.emplace_front(
            gauge_equation_index,
            magnetic_vector_potential_x_unknown_index_neighbor1,
            (y_neighbor2 - y) / denominator);
        A_triplets.emplace_front(
            gauge_equation_index,
            magnetic_vector_potential_y_unknown_index_neighbor1,
            (x - x_neighbor2) / denominator);
        A_triplets.emplace_front(
            gauge_equation_index,
            magnetic_vector_potential_x_unknown_index_neighbor2,
            (y - y_neighbor1) / denominator);
        A_triplets.emplace_front(
            gauge_equation_index,
            magnetic_vector_potential_y_unknown_index_neighbor2,
            (x_neighbor1 - x) / denominator);
        A_triplets.emplace_front(gauge_equation_index,
                                 magnetic_vector_potential_x_unknown_index,
                                 (y_neighbor1 - y_neighbor2) / denominator);
        A_triplets.emplace_front(gauge_equation_index,
                                 magnetic_vector_potential_y_unknown_index,
                                 (x_neighbor2 - x_neighbor1) / denominator);
      }

      // Set the coefficients for the electric fields and magnetic vector
      // potentials for the equations corresponding to Faraday's law.
      A_triplets.emplace_front(faraday_law_x_equation_index,
                               electric_field_x_unknown_index, 1);
      A_triplets.emplace_front(faraday_law_x_equation_index,
                               magnetic_vector_potential_x_unknown_index,
                               std::complex<double>(0, omega));
      A_triplets.emplace_front(faraday_law_y_equation_index,
                               electric_field_y_unknown_index, 1);
      A_triplets.emplace_front(faraday_law_y_equation_index,
                               magnetic_vector_potential_y_unknown_index,
                               std::complex<double>(0, omega));

      // Set the coefficients for the electric fields for the equations
      // corresponding to Ampere's law.
      A_triplets.emplace_front(
          ampere_law_x_equation_index, electric_field_x_unknown_index,
          conductor_properties.permeability(config_.frequency()) *
              conductor_properties.conductivity(config_.frequency()) *
              num_adjacent_triangles);
      A_triplets.emplace_front(
          ampere_law_y_equation_index, electric_field_y_unknown_index,
          conductor_properties.permeability(config_.frequency()) *
              conductor_properties.conductivity(config_.frequency()) *
              num_adjacent_triangles);
    }

    // Set the voltage boundary conditions for the boundary nodes. The field
    // equations at the interface are provided by the adjacent insulator
    // region.
    const auto conductor_boundary_node_tags =
        GetNodes(conductor_tag, dimension(), gmsh::NodeType::kBoundary);
    for (const auto tag : conductor_boundary_node_tags) {
      const auto gauss_law_electric_potential_equation_index =
          this->gauss_law_electric_potential_equation_index(tag);

      const auto electric_potential_unknown_index =
          this->electric_potential_unknown_index(tag);

      // Set the voltage boundary conditions.
      A_triplets.emplace_front(gauss_law_electric_potential_equation_index,
                               electric_potential_unknown_index, 1);
      b_triplets.emplace_front(gauss_law_electric_potential_equation_index, 0,
                               GetAcPhasor(conductor_tag));
    }
    boundary_tags.insert(conductor_boundary_node_tags.cbegin(),
                         conductor_boundary_node_tags.cend());
  }

  // Fill in the equations for the nodes within the insulators, including the
  // boundary nodes.
  for (const auto insulator_tag : insulator_entity_tags) {
    const auto insulator_properties =
        model::MaterialProperties::kMaterialToProperties.at(
            GetMaterialForEntity(insulator_tag));
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
      const auto gauss_law_electric_potential_equation_index =
          this->gauss_law_electric_potential_equation_index(tag);
      const auto faraday_law_x_equation_index =
          this->faraday_law_x_equation_index(tag);
      const auto faraday_law_y_equation_index =
          this->faraday_law_y_equation_index(tag);
      const auto ampere_law_x_equation_index =
          this->ampere_law_x_equation_index(tag);
      const auto ampere_law_y_equation_index =
          this->ampere_law_y_equation_index(tag);
      const auto gauge_equation_index = this->gauge_equation_index(tag);

      // Unknown indices.
      const auto electric_potential_unknown_index =
          this->electric_potential_unknown_index(tag);
      const auto electric_field_x_unknown_index =
          this->electric_field_x_unknown_index(tag);
      const auto electric_field_y_unknown_index =
          this->electric_field_y_unknown_index(tag);
      const auto magnetic_vector_potential_x_unknown_index =
          this->magnetic_vector_potential_x_unknown_index(tag);
      const auto magnetic_vector_potential_y_unknown_index =
          this->magnetic_vector_potential_y_unknown_index(tag);
      const auto magnetic_flux_density_z_unknown_index =
          this->magnetic_flux_density_z_unknown_index(tag);
      const auto gauge_unknown_index = this->gauge_unknown_index(tag);

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
        const auto magnetic_vector_potential_x_unknown_index_neighbor1 =
            this->magnetic_vector_potential_x_unknown_index(tag_neighbor1);
        const auto magnetic_vector_potential_x_unknown_index_neighbor2 =
            this->magnetic_vector_potential_x_unknown_index(tag_neighbor2);
        const auto magnetic_vector_potential_y_unknown_index_neighbor1 =
            this->magnetic_vector_potential_y_unknown_index(tag_neighbor1);
        const auto magnetic_vector_potential_y_unknown_index_neighbor2 =
            this->magnetic_vector_potential_y_unknown_index(tag_neighbor2);
        const auto magnetic_flux_density_z_unknown_index_neighbor1 =
            this->magnetic_flux_density_z_unknown_index(tag_neighbor1);
        const auto magnetic_flux_density_z_unknown_index_neighbor2 =
            this->magnetic_flux_density_z_unknown_index(tag_neighbor2);
        const auto gauge_unknown_index_neighbor1 =
            this->gauge_unknown_index(tag_neighbor1);
        const auto gauge_unknown_index_neighbor2 =
            this->gauge_unknown_index(tag_neighbor2);

        // Calculate the denominator.
        const auto denominator = (x_neighbor1 * y_neighbor2 - x_neighbor1 * y -
                                  x_neighbor2 * y_neighbor1 + x_neighbor2 * y +
                                  x * y_neighbor1 - x * y_neighbor2);

        // Add the coefficients for the equation corresponding to Gauss's law.
        if (!boundary_tags.contains(tag)) {
          A_triplets.emplace_front(gauss_law_electric_potential_equation_index,
                                   electric_field_x_unknown_index_neighbor1,
                                   (y_neighbor2 - y) / denominator);
          A_triplets.emplace_front(gauss_law_electric_potential_equation_index,
                                   electric_field_y_unknown_index_neighbor1,
                                   (x - x_neighbor2) / denominator);
          A_triplets.emplace_front(gauss_law_electric_potential_equation_index,
                                   electric_field_x_unknown_index_neighbor2,
                                   (y - y_neighbor1) / denominator);
          A_triplets.emplace_front(gauss_law_electric_potential_equation_index,
                                   electric_field_y_unknown_index_neighbor2,
                                   (x_neighbor1 - x) / denominator);
          A_triplets.emplace_front(gauss_law_electric_potential_equation_index,
                                   electric_field_x_unknown_index,
                                   (y_neighbor1 - y_neighbor2) / denominator);
          A_triplets.emplace_front(gauss_law_electric_potential_equation_index,
                                   electric_field_y_unknown_index,
                                   (x_neighbor2 - x_neighbor1) / denominator);
        }

        // Add the coefficients for the equation corresponding to Faraday's law
        // in the x-direction.
        A_triplets.emplace_front(faraday_law_x_equation_index,
                                 electric_potential_unknown_index_neighbor1,
                                 (y_neighbor2 - y) / denominator);
        A_triplets.emplace_front(faraday_law_x_equation_index,
                                 electric_potential_unknown_index_neighbor2,
                                 (y - y_neighbor1) / denominator);
        A_triplets.emplace_front(faraday_law_x_equation_index,
                                 electric_potential_unknown_index,
                                 (y_neighbor1 - y_neighbor2) / denominator);

        // Add the coefficients for the equation corresponding to Faraday's law
        // in the y-direction.
        A_triplets.emplace_front(faraday_law_y_equation_index,
                                 electric_potential_unknown_index_neighbor1,
                                 (x - x_neighbor2) / denominator);
        A_triplets.emplace_front(faraday_law_y_equation_index,
                                 electric_potential_unknown_index_neighbor2,
                                 (x_neighbor1 - x) / denominator);
        A_triplets.emplace_front(faraday_law_y_equation_index,
                                 electric_potential_unknown_index,
                                 (x_neighbor2 - x_neighbor1) / denominator);

        // Add the coefficients for the equation corresponding to Ampere's law
        // in the x-direction.
        A_triplets.emplace_front(
            ampere_law_x_equation_index,
            magnetic_flux_density_z_unknown_index_neighbor1,
            -(x - x_neighbor2) / denominator);
        A_triplets.emplace_front(
            ampere_law_x_equation_index,
            magnetic_flux_density_z_unknown_index_neighbor2,
            -(x_neighbor1 - x) / denominator);
        A_triplets.emplace_front(ampere_law_x_equation_index,
                                 magnetic_flux_density_z_unknown_index,
                                 -(x_neighbor2 - x_neighbor1) / denominator);
        A_triplets.emplace_front(ampere_law_x_equation_index,
                                 gauge_unknown_index_neighbor1,
                                 (y_neighbor2 - y) / denominator);
        A_triplets.emplace_front(ampere_law_x_equation_index,
                                 gauge_unknown_index_neighbor2,
                                 (y - y_neighbor1) / denominator);
        A_triplets.emplace_front(ampere_law_x_equation_index,
                                 gauge_unknown_index,
                                 (y_neighbor1 - y_neighbor2) / denominator);

        // Add the coefficients for the equation corresponding to Ampere's law
        // in the y-direction.
        A_triplets.emplace_front(
            ampere_law_y_equation_index,
            magnetic_flux_density_z_unknown_index_neighbor1,
            (y_neighbor2 - y) / denominator);
        A_triplets.emplace_front(
            ampere_law_y_equation_index,
            magnetic_flux_density_z_unknown_index_neighbor2,
            (y - y_neighbor1) / denominator);
        A_triplets.emplace_front(ampere_law_y_equation_index,
                                 magnetic_flux_density_z_unknown_index,
                                 (y_neighbor1 - y_neighbor2) / denominator);
        A_triplets.emplace_front(ampere_law_y_equation_index,
                                 gauge_unknown_index_neighbor1,
                                 (x - x_neighbor2) / denominator);
        A_triplets.emplace_front(ampere_law_y_equation_index,
                                 gauge_unknown_index_neighbor2,
                                 (x_neighbor1 - x) / denominator);
        A_triplets.emplace_front(ampere_law_y_equation_index,
                                 gauge_unknown_index,
                                 (x_neighbor2 - x_neighbor1) / denominator);

        // Add the coefficients for the equation corresponding to the gauge.
        A_triplets.emplace_front(
            gauge_equation_index,
            magnetic_vector_potential_x_unknown_index_neighbor1,
            (y_neighbor2 - y) / denominator);
        A_triplets.emplace_front(
            gauge_equation_index,
            magnetic_vector_potential_y_unknown_index_neighbor1,
            (x - x_neighbor2) / denominator);
        A_triplets.emplace_front(
            gauge_equation_index,
            magnetic_vector_potential_x_unknown_index_neighbor2,
            (y - y_neighbor1) / denominator);
        A_triplets.emplace_front(
            gauge_equation_index,
            magnetic_vector_potential_y_unknown_index_neighbor2,
            (x_neighbor1 - x) / denominator);
        A_triplets.emplace_front(gauge_equation_index,
                                 magnetic_vector_potential_x_unknown_index,
                                 (y_neighbor1 - y_neighbor2) / denominator);
        A_triplets.emplace_front(gauge_equation_index,
                                 magnetic_vector_potential_y_unknown_index,
                                 (x_neighbor2 - x_neighbor1) / denominator);
      }

      // Set the coefficients for the electric fields and magnetic vector
      // potentials for the equations corresponding to Faraday's law.
      A_triplets.emplace_front(faraday_law_x_equation_index,
                               electric_field_x_unknown_index,
                               num_adjacent_triangles);
      A_triplets.emplace_front(
          faraday_law_x_equation_index,
          magnetic_vector_potential_x_unknown_index,
          std::complex<double>(0, omega * num_adjacent_triangles));
      A_triplets.emplace_front(faraday_law_y_equation_index,
                               electric_field_y_unknown_index,
                               num_adjacent_triangles);
      A_triplets.emplace_front(
          faraday_law_y_equation_index,
          magnetic_vector_potential_y_unknown_index,
          std::complex<double>(0, omega * num_adjacent_triangles));

      // Set the coefficients for the electric fields for the equations
      // corresponding to Ampere's law.
      A_triplets.emplace_front(
          ampere_law_x_equation_index, electric_field_x_unknown_index,
          std::complex<double>(
              insulator_properties.permeability(config_.frequency()) *
                  insulator_properties.conductivity(config_.frequency()) *
                  num_adjacent_triangles,
              omega * insulator_properties.permeability(config_.frequency()) *
                  insulator_properties.permittivity(config_.frequency()) *
                  num_adjacent_triangles));
      A_triplets.emplace_front(
          ampere_law_y_equation_index, electric_field_y_unknown_index,
          std::complex<double>(
              insulator_properties.permeability(config_.frequency()) *
                  insulator_properties.conductivity(config_.frequency()) *
                  num_adjacent_triangles,
              omega * insulator_properties.permeability(config_.frequency()) *
                  insulator_properties.permittivity(config_.frequency()) *
                  num_adjacent_triangles));
    }
  }

  // Create the sparse matrix-vector equation.
  Eigen::SparseMatrix<std::complex<double>> A(num_unknowns(), num_unknowns());
  A.setFromTriplets(A_triplets.cbegin(), A_triplets.cend());
  Eigen::VectorXcd b = Eigen::VectorXcd::Zero(num_unknowns());
  for (const auto& triplet : b_triplets) {
    b(triplet.row()) += triplet.value();
  }

  // Solve for the electromagnetic fields.
  Eigen::SuperLU<Eigen::SparseMatrix<std::complex<double>>> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(absl::StrFormat(
        "Failed to compute the decomposition of the matrix: %d.",
        solver.info()));
  }
  const Eigen::VectorXcd x = solver.solve(b);
  electric_potential_ = x(Eigen::seqN(0, num_electric_potential_unknowns()));
  electric_field_ = x(Eigen::seqN(num_electric_potential_unknowns(),
                                  num_electric_field_unknowns()))
                        .reshaped<Eigen::RowMajor>(num_nodes_, dimension());
  magnetic_vector_potential_ =
      x(Eigen::seqN(
            num_electric_potential_unknowns() + num_electric_field_unknowns(),
            num_magnetic_vector_potential_unknowns()))
          .reshaped<Eigen::RowMajor>(num_nodes_, dimension());
  magnetic_flux_density_ =
      x(Eigen::seqN(num_electric_potential_unknowns() +
                        num_electric_field_unknowns() +
                        num_magnetic_vector_potential_unknowns(),
                    num_magnetic_flux_density_unknowns()))
          .reshaped<Eigen::RowMajor>(num_nodes_, 1);
}

}  // namespace solver
