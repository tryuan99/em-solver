import google.protobuf
from absl import app, flags
from proto.mesh_config_pb2 import MeshConfig

from mesh.mesh_generator import MeshGenerator2D

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    # Parse the mesh generator configuration.
    with open(FLAGS.mesh_generator_config, "r") as mesh_generator_config:
        mesh_config = google.protobuf.text_format.Parse(
            mesh_generator_config.read(), MeshConfig())

    # Parse the entity configuration.
    with open(FLAGS.entity_config, "r") as entity_config:
        mesh_config = google.protobuf.text_format.Merge(entity_config.read(),
                                                        mesh_config)

    # Generate the mesh.
    mesh_generator = MeshGenerator2D(FLAGS.input_file, mesh_config)
    if FLAGS.mesh_output is not None:
        mesh_generator.write_mesh_file(FLAGS.mesh_output)
    if FLAGS.launch:
        mesh_generator.launch()


if __name__ == "__main__":
    flags.DEFINE_string("input_file",
                        "cad/capacitor/capacitor_three_plates_2d.geo",
                        "Input file.")
    flags.DEFINE_string("mesh_generator_config",
                        "mesh/configs/mesh_generator_config_default.pbtxt",
                        "Mesh generator configuration file.")
    flags.DEFINE_string(
        "entity_config",
        "mesh/configs/capacitor_mesh_config_three_plates.pbtxt",
        "Entity configuration file.")
    flags.DEFINE_string("mesh_output", None, "Mesh output file.")
    flags.DEFINE_bool("launch", True, "If true, launch the GUI.")

    app.run(main)
