from absl import app, flags

FLAGS = flags.FLAGS

from solver.capacitor_three_plates_electrostatic_solver import \
    CapacitorThreePlatesElectrostaticSolver2D


def main(argv):
    assert len(argv) == 1

    capacitor_solver = CapacitorThreePlatesElectrostaticSolver2D(
        FLAGS.mesh_file)
    capacitor_solver.solve()
    capacitor_solver.plot_voltage()
    capacitor_solver.plot_electric_field()


if __name__ == "__main__":
    flags.DEFINE_string("mesh_file", None, "Mesh file.")
    flags.mark_flag_as_required("mesh_file")

    app.run(main)
