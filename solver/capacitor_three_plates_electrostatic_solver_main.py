import google.protobuf
from absl import app, flags
from proto.solver_config_pb2 import SolverConfig

from solver.capacitor_three_plates_electrostatic_solver import \
    CapacitorThreePlatesElectrostaticSolver2D

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    # Parse the solver generator configuration.
    with open(FLAGS.solver_config, "r") as solver_config_file:
        solver_config = google.protobuf.text_format.Parse(
            solver_config_file.read(), SolverConfig())

    # Solve the mesh.
    capacitor_solver = CapacitorThreePlatesElectrostaticSolver2D(
        FLAGS.mesh_file, solver_config)
    capacitor_solver.solve()
    capacitor_solver.plot_voltage()
    capacitor_solver.plot_electric_field()


if __name__ == "__main__":
    flags.DEFINE_string("mesh_file", None, "Mesh file.")
    flags.DEFINE_string("solver_config", None, "Solver configuration file.")
    flags.mark_flags_as_required(["mesh_file", "solver_config"])

    app.run(main)
