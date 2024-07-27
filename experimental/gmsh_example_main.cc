#include <gmsh.h>

#include <cstdlib>
#include <vector>

#include "base/base.h"
#include "base/commandlineflags.h"

DEFINE_string(mesh_file, "", "Mesh file.");
DEFINE_uint32(dimension, 2, "Dimension of the mesh structure.");

int main(int argc, char** argv) {
  base::Init(argc, argv);

  // Initialize Gmsh.
  gmsh::initialize();

  // Open the mesh file.
  gmsh::open(FLAGS(mesh_file));

  // Log the number of nodes in the mesh.
  std::vector<std::size_t> node_tags;
  std::vector<double> node_coordinates;
  std::vector<double> node_parametric_coordinates;
  gmsh::model::mesh::getNodes(node_tags, node_coordinates,
                              node_parametric_coordinates, FLAGS(dimension),
                              /*tag=*/-1);
  LOG(INFO) << "Number of nodes: " << node_tags.size() << ".";

  return EXIT_SUCCESS;
}
