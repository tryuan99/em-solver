from absl import app, flags

from visualization.electromagnetic_field_plotter import \
    ElectromagneticFieldPlotter2D

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    plotter = ElectromagneticFieldPlotter2D(FLAGS.mesh_file, FLAGS.csv_file)
    plotter.plot_electric_potential()
    plotter.plot_electric_field()
    plotter.plot_magnetic_vector_potential()
    plotter.plot_magnetic_flux_density()


if __name__ == "__main__":
    flags.DEFINE_string("mesh_file", None, "Mesh file.")
    flags.DEFINE_string("csv_file", None, "CSV solution file.")
    flags.mark_flags_as_required(["mesh_file", "csv_file"])

    app.run(main)
