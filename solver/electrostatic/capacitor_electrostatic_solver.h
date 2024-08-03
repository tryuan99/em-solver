// The capacitor electrostatic solver solves for the electrostatic field around
// a capacitor.

#pragma once

#include <string>

#include "proto/solver_config.pb.h"
#include "solver/electrostatic/electrostatic_solver.h"

namespace solver {

// 2D capacitor electrostatic solver.
class CapacitorElectrostaticSolver2D : public ElectrostaticSolver2D {
 public:
  CapacitorElectrostaticSolver2D(const std::string& mesh_file,
                                 const SolverConfig solver_config)
      : ElectrostaticSolver2D(mesh_file, std::move(solver_config)) {}

  // Calculate the capacitance.
  double CalculateCapacitance() const;

 protected:
  // Validate the mesh.
  void ValidateMesh() const override;
};

}  // namespace solver
