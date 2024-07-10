import google.protobuf
from absl import app, flags, logging
from proto.solver_config_pb2 import SolverConfig

from solver.electrodynamic.capacitor_electrodynamic_solver import \
    CapacitorElectrodynamicSolver2D
from visualization.electromagnetic_field_plotter import \
    ElectromagneticFieldPlotter2D

FLAGS = flags.FLAGS


def _solve_mesh(mesh_file: str, solver_config: str, csv_output: str) -> None:
    """Solves the mesh.

    Args:
        mesh_file: Mesh file.
        solver_config: Solver configuration.
        csv_output: CSV output file.
    """
    # Parse the solver configuration.
    with open(solver_config, "r") as solver_config_file:
        solver_config = google.protobuf.text_format.Parse(
            solver_config_file.read(), SolverConfig())

    # Solve the mesh.
    capacitor_solver = CapacitorElectrodynamicSolver2D(mesh_file, solver_config)
    capacitor_solver.solve()
    if csv_output is not None:
        capacitor_solver.write_solution(csv_output)


def _plot_electromagnetic_fields(mesh_file: str, csv_output: str) -> None:
    """Plots the electromagnetic fields.

    Args:
        mesh_file: Mesh file.
        csv_output: CSV output file.
    """
    plotter = ElectromagneticFieldPlotter2D(mesh_file, csv_output)
    plotter.plot_electric_potential()
    plotter.plot_electric_field()
    plotter.plot_magnetic_vector_potential()
    plotter.plot_magnetic_flux_density()


def main(argv):
    assert len(argv) == 1

    _solve_mesh(FLAGS.mesh_file, FLAGS.solver_config, FLAGS.csv_output)
    if FLAGS.csv_output is not None and FLAGS.plot:
        _plot_electromagnetic_fields(FLAGS.mesh_file, FLAGS.csv_output)


if __name__ == "__main__":
    flags.DEFINE_string("mesh_file", None, "Mesh file.")
    flags.DEFINE_string("solver_config", None, "Solver configuration file.")
    flags.DEFINE_string("csv_output", None, "CSV output file.")
    flags.DEFINE_boolean("plot", True,
                         "If true, plot the electromagnetic fields.")
    flags.mark_flags_as_required(["mesh_file", "solver_config"])

    app.run(main)
