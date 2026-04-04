// The capacitor electrodynamic solver solves for the electromagnetic field
// around a capacitor.

#pragma once

#include <string>

#include "proto/solver_config.pb.h"
#include "solver/electrodynamic/electrodynamic_solver.h"

namespace solver {

// 2D capacitor electrodynamic solver.
class CapacitorElectrodynamicSolver2D : public ElectrodynamicSolver2D {
 public:
  CapacitorElectrodynamicSolver2D(const std::string& mesh_file,
                                  const SolverConfig solver_config)
      : ElectrodynamicSolver2D(mesh_file, std::move(solver_config)) {
    EnsureMeshValidated();
  }

 protected:
  // Validate the mesh.
  void ValidateMesh() const override;
};

}  // namespace solver
