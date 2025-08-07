#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "base/base.h"
#include "base/commandlineflags.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "solver/electrostatic/capacitor_electrostatic_solver.h"

DEFINE_string(mesh_file, "", "Mesh file.");
DEFINE_string(solver_config, "", "Solver configuration file.");
DEFINE_string(csv_output, "", "CSV output file.");

int main(int argc, char** argv) {
  base::Init(argc, argv);

  // Parse the solver configuration.
  std::ifstream ifs(FLAGS(solver_config).c_str());
  if (!ifs.is_open()) {
    throw std::runtime_error(
        "Unable to open the output solver configuration file.");
  }
  google::protobuf::io::IstreamInputStream file_stream(&ifs);
  solver::SolverConfig solver_config;
  if (!google::protobuf::TextFormat::Parse(&file_stream, &solver_config)) {
    throw std::runtime_error("Failed to parse the solver configuration file.");
  }

  // Solve the mesh.
  solver::CapacitorElectrostaticSolver2D capacitor_solver(FLAGS(mesh_file),
                                                          solver_config);
  capacitor_solver.Solve();
  if (!FLAGS(csv_output).empty()) {
    capacitor_solver.WriteSolution(FLAGS(csv_output));
  }

  return EXIT_SUCCESS;
}
