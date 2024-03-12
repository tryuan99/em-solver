from absl import app, flags

FLAGS = flags.FLAGS

from mesh_generator.capacitor_mesh_generator import CapacitorMeshGenerator3D


def main(argv):
    assert len(argv) == 1

    capacitor_mesh_generator = CapacitorMeshGenerator3D(FLAGS.input_file)
    capacitor_mesh_generator.write_mesh_file(FLAGS.mesh_output)


if __name__ == "__main__":
    flags.DEFINE_string("input_file",
                        "cad/capacitor/capacitor_m7_m8_100um_100um.step",
                        "Input file.")
    flags.DEFINE_string("mesh_output", None, "Mesh output file.")
    flags.mark_flag_as_required("mesh_output")

    app.run(main)
