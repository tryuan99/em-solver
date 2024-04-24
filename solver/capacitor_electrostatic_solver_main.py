from absl import app, flags, logging

FLAGS = flags.FLAGS

from solver.capacitor_electrostatic_solver import \
    CapacitorElectrostaticSolver2D


def main(argv):
    assert len(argv) == 1

    capacitor_solver = CapacitorElectrostaticSolver2D(FLAGS.mesh_file)
    capacitor_solver.solve()
    capacitor_solver.plot_voltage()
    capacitor_solver.plot_electric_field()
    capacitance = capacitor_solver.calculate_capacitance()
    logging.info("Capacitance = %f", capacitance)


if __name__ == "__main__":
    flags.DEFINE_string("mesh_file", None, "Mesh file.")
    flags.mark_flag_as_required("mesh_file")

    app.run(main)
