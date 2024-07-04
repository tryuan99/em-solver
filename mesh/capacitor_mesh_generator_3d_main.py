import google.protobuf
from absl import app, flags

FLAGS = flags.FLAGS

from proto.mesh_config_pb2 import MeshConfig

from mesh.capacitor_mesh_generator import CapacitorMeshGenerator3D


def main(argv):
    assert len(argv) == 1

    # Parse the mesh generator configuration.
    with open(FLAGS.mesh_generator_config, "r") as mesh_generator_config:
        mesh_config = google.protobuf.text_format.Parse(
            mesh_generator_config.read(), MeshConfig())

    # Parse the capacitor mesh configuration.
    with open(FLAGS.capacitor_mesh_config, "r") as capacitor_mesh_config:
        mesh_config = google.protobuf.text_format.Merge(
            capacitor_mesh_config.read(), mesh_config)

    # Generate the mesh.
    capacitor_mesh_generator = CapacitorMeshGenerator3D(FLAGS.input_file,
                                                        mesh_config)
    if FLAGS.mesh_output is not None:
        capacitor_mesh_generator.write_mesh_file(FLAGS.mesh_output)
    if FLAGS.launch:
        capacitor_mesh_generator.launch()


if __name__ == "__main__":
    flags.DEFINE_string("input_file",
                        "cad/capacitor/capacitor_m7_m8_100um_100um.step",
                        "Input file.")
    flags.DEFINE_string("mesh_generator_config",
                        "mesh/configs/mesh_generator_config_default.pbtxt",
                        "Mesh generator configuration file.")
    flags.DEFINE_string("capacitor_mesh_config",
                        "mesh/configs/capacitor_mesh_config_default.pbtxt",
                        "Capacitor mesh configuration file.")
    flags.DEFINE_string("mesh_output", None, "Mesh output file.")
    flags.DEFINE_bool("launch", True, "If true, launch the GUI.")

    app.run(main)
