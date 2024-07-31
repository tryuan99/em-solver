#include "solver/electrostatic/capacitor_electrostatic_solver.h"

#include <stdexcept>

#include "absl/strings/str_format.h"
#include "model/material.h"
#include "proto/capacitor.pb.h"

namespace solver {

double CapacitorElectrostaticSolver2D::CalculateCapacitance() const {
  // TODO(titan): Implement calculating the capacitance.
  return 0;
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
