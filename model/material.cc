#include "model/material.h"

#include <limits>
#include <unordered_map>

#include "proto/material.pb.h"

namespace model {

std::unordered_map<Material, MaterialProperties>
    MaterialProperties::kMaterialToProperties{
        // Ideal materials.
        {Material::INSULATOR,
         MaterialProperties(
             Material::INSULATOR, /*relative_permittivity=*/1,
             /*relative_permeability=*/1,
             /*resistivity=*/std::numeric_limits<double>::infinity())},
        {Material::CONDUCTOR,
         MaterialProperties(
             Material::CONDUCTOR,
             /*relative_permittivity=*/std::numeric_limits<double>::infinity(),
             /*relative_permeability=*/1, /*resistivity=*/0)},
        // Insulators.
        {Material::AIR,
         MaterialProperties(Material::AIR, /*relative_permittivity=*/1.0006,
                            /*relative_permeability=*/1,
                            /*resistivity=*/1e13)},
        {Material::SILICON_DIOXIDE,
         MaterialProperties(Material::SILICON_DIOXIDE,
                            /*relative_permittivity=*/3.9,
                            /*relative_permeability=*/1,
                            /*resistivity=*/1e15)},
        // Conductors.
        {Material::COPPER,
         MaterialProperties(
             Material::COPPER,
             /*relative_permittivity=*/std::numeric_limits<double>::infinity(),
             /*relative_permeability=*/1, /*resistivity=*/1.68e-8)},
        {Material::GOLD,
         MaterialProperties(
             Material::GOLD,
             /*relative_permittivity=*/std::numeric_limits<double>::infinity(),
             /*relative_permeability=*/1, /*resistivity=*/2.44e-8)},
    };

}
