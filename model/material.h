// The material model defines the properties of various materials, including
// conductors and insulators.

#pragma once

#include <cstdbool>
#include <unordered_map>

#include "model/constants.h"
#include "proto/material.pb.h"

namespace model {

// Conductivity threshold for a conductor in S/m.
constexpr double kConductorConductivityThreshold = 1e-2;

// Conductivity threshold for an insulator in S/m.
constexpr double kInsulatorConductivityThreshold = 1e-4;

// Material properties.
class MaterialProperties {
 public:
  // Map from various materials to their properties.
  static std::unordered_map<Material, MaterialProperties> kMaterialToProperties;

  MaterialProperties(const Material material,
                     const double relative_permittivity,
                     const double relative_permeability,
                     const double resistivity)
      : material_(material),
        relative_permittivity_(relative_permittivity),
        relative_permeability_(relative_permeability),
        resistivity_(resistivity) {}

  // Get the permittivity at the given frequency in F/m.
  double permittivity(const double frequency = 0) const {
    return relative_permittivity_ * kVacuumPermittivity;
  }

  // Get the permeability at the given frequency in H/m.
  double permeability(const double frequency = 0) const {
    return relative_permeability_ * kVacuumPermeability;
  }

  // Get the conductivity at the given frequency in S/m.
  double conductivity(const double frequency = 0) const {
    return 1 / resistivity_;
  }

  // Get the resistivity at the given frequency in Ohm*m.
  double resistivity(const double frequency = 0) const { return resistivity_; }

  // Return whether the given material is a conductor.
  bool is_conductor() const {
    return conductivity(/*frequency=*/0) > kConductorConductivityThreshold;
  }

  // Return whether the given material is an insulator.
  bool is_insulator() const {
    return conductivity(/*frequency=*/0) < kInsulatorConductivityThreshold;
  }

 private:
  // Material.
  Material material_ = Material::UNSPECIFIED;

  // Relative permittivity.
  double relative_permittivity_ = 0;

  // Relative permeability.
  double relative_permeability_ = 0;

  // Resistivity in Ohm*m.
  double resistivity_ = 0;
};

}  // namespace model
