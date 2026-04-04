#include "solver/electrostatic/capacitor_electrostatic_solver.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "absl/strings/str_format.h"
#include "mesh/gmsh_interface.h"
#include "model/material.h"
#include "proto/capacitor.pb.h"

namespace solver {

double CapacitorElectrostaticSolver2D::CalculateCapacitance() const {
  if (!solved_) {
    throw std::logic_error(
        "The electrostatic fields must be solved before calculating "
        "capacitance.");
  }

  const auto ground_plate_boundary_node_tags =
      GetNodes(model::CapacitorEntity::GROUND_PLATE, dimension(),
               gmsh::NodeType::kBoundary);
  if (ground_plate_boundary_node_tags.empty()) {
    throw std::runtime_error("Ground plate has no boundary nodes.");
  }

  const auto node_tag_to_coordinates =
      GetNodeCoordinates(/*tag=*/-1, dimension());

  double center_x = 0;
  double center_y = 0;
  for (const auto tag : ground_plate_boundary_node_tags) {
    const auto& coordinates = node_tag_to_coordinates.at(tag);
    center_x += coordinates[0];
    center_y += coordinates[1];
  }
  center_x /= ground_plate_boundary_node_tags.size();
  center_y /= ground_plate_boundary_node_tags.size();

  const auto adjacencies =
      GetAdjacencies(dimension(), model::CapacitorEntity::GROUND_PLATE);
  const auto& line_tags = adjacencies.second;
  if (line_tags.empty()) {
    throw std::runtime_error("Ground plate has no boundary lines.");
  }

  std::complex<double> electric_flux = 0;
  for (const auto line_tag : line_tags) {
    for (const auto& [_, line_node_tags] : GetLines(line_tag)) {
      const auto& node_coordinates_neighbor1 =
          node_tag_to_coordinates.at(line_node_tags[0]);
      const auto& node_coordinates_neighbor2 =
          node_tag_to_coordinates.at(line_node_tags[1]);
      const auto line_vector_x =
          node_coordinates_neighbor2[0] - node_coordinates_neighbor1[0];
      const auto line_vector_y =
          node_coordinates_neighbor2[1] - node_coordinates_neighbor1[1];
      const auto neighbor1_to_center_x =
          center_x - node_coordinates_neighbor1[0];
      const auto neighbor1_to_center_y =
          center_y - node_coordinates_neighbor1[1];
      const auto cross_product = line_vector_x * neighbor1_to_center_y -
                                 line_vector_y * neighbor1_to_center_x;
      const auto normal_sign = cross_product >= 0 ? -1.0 : 1.0;
      auto normal_vector_x = normal_sign * -line_vector_y;
      auto normal_vector_y = normal_sign * line_vector_x;
      const auto line_length = std::hypot(normal_vector_x, normal_vector_y);
      normal_vector_x /= line_length;
      normal_vector_y /= line_length;

      const auto electric_field_neighbor1 =
          electric_field_.row(node_index_from_tag(line_node_tags[0]));
      const auto electric_field_neighbor2 =
          electric_field_.row(node_index_from_tag(line_node_tags[1]));
      const auto electric_field_averaged =
          (electric_field_neighbor1 + electric_field_neighbor2) / 2.0;
      electric_flux +=
          line_length * (electric_field_averaged(0) * normal_vector_x +
                         electric_field_averaged(1) * normal_vector_y);
    }
  }

  const auto dielectric_material =
      GetMaterialForEntity(model::CapacitorEntity::DIELECTRIC);
  const auto dielectric_permittivity =
      model::MaterialProperties::kMaterialToProperties.at(dielectric_material)
          .permittivity();
  const auto charge = electric_flux.real() * dielectric_permittivity;
  return std::abs(charge / GetDcVoltage(model::CapacitorEntity::VDD_PLATE));
}

void CapacitorElectrostaticSolver2D::ValidateMesh() const {
  // Validate the materials of the capacitor plates and the dielectric.
  const std::vector<model::CapacitorEntity> capacitor_plate_entity_tags{
      model::CapacitorEntity::GROUND_PLATE, model::CapacitorEntity::VDD_PLATE};
  for (const auto conductor_entity_tag : capacitor_plate_entity_tags) {
    if (!model::MaterialProperties::kMaterialToProperties
             .at(GetMaterialForEntity(conductor_entity_tag))
             .is_conductor()) {
      throw std::invalid_argument(absl::StrFormat(
          "Entity %d is not a conductor.", conductor_entity_tag));
    }
  }
  if (!model::MaterialProperties::kMaterialToProperties
           .at(GetMaterialForEntity(model::CapacitorEntity::DIELECTRIC))
           .is_insulator()) {
    throw std::invalid_argument(absl::StrFormat(
        "Entity %d is not an insulator.", model::CapacitorEntity::DIELECTRIC));
  }
}

}  // namespace solver
