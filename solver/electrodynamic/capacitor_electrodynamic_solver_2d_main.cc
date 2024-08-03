#include <fcntl.h>

#include <cstdlib>

#include "base/base.h"
#include "base/commandlineflags.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "solver/electrodynamic/capacitor_electrodynamic_solver.h"

DEFINE_string(mesh_file, "", "Mesh file.");
DEFINE_string(solver_config, "", "Solver configuration file.");
DEFINE_string(csv_output, "", "CSV output file.");

int main(int argc, char** argv) {
  base::Init(argc, argv);

  // Parse the solver configuration.
  int fd = open(FLAGS(solver_config).c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error(
        "Unable to open the output solver configuration file.");
  }
  google::protobuf::io::FileInputStream solver_config_file(fd);
  solver::SolverConfig solver_config;
  if (!google::protobuf::TextFormat::Parse(&solver_config_file,
                                           &solver_config)) {
    throw std::runtime_error("Failed to parse the solver configuration file.");
  }
  solver_config_file.Close();

  // Solve the mesh.
  solver::CapacitorElectrodynamicSolver2D capacitor_solver(FLAGS(mesh_file),
                                                           solver_config);
  capacitor_solver.Solve();
  if (!FLAGS(csv_output).empty()) {
    capacitor_solver.WriteSolution(FLAGS(csv_output));
  }

  return EXIT_SUCCESS;
}
