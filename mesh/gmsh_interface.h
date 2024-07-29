// The Gmsh interface class provides an interface to gmsh utilities.

#pragma once

#include <cstdlib>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace gmsh {

// Type definitions.
using Tag = std::size_t;
using Coordinates = std::tuple<double, double, double>;
using DimTag = std::pair<int, int>;

// Mesh element type enumeration.
enum class ElementType {
  kTwoNodeLine = 1,
  kThreeNodeTriangle = 2,
  kFourNodeQuadrangle = 3,
  kFourNodeTetrahedron = 4,
  kEightNodeHexahedron = 5,
  kSixNodePrism = 6,
  kFiveNodePyramid = 7,
  kOneNodePoint = 15,
};

// Node type enumeration.
enum class NodeType {
  kAll = 0,
  kInternal,
  kBoundary,
};

// Gmsh interface.
class GmshInterface {
 public:
  GmshInterface();
  ~GmshInterface();

  // Write the mesh to the output file.
  static void WriteMeshFile(const std::string& mesh_file);

  // Launch the GUI.
  static void LaunchGui();

  // Get the list of nodes for the tag and dimension and return the list of node
  // tags. The node tags may be duplicated across multiple dimensions.
  static std::vector<Tag> GetNodes(int tag = -1, int dim = -1,
                                   NodeType node_type = NodeType::kAll);

  // Get the node coordinates and return a map from the node tags to the node
  // coordinates.
  static std::unordered_map<Tag, Coordinates> GetNodeCoordinates(int tag = -1,
                                                                 int dim = -1);

  // Get the boundary entities of the given entity and return a list of pairs
  // containing the dimension and tag of each bounding entity.
  static std::vector<DimTag> GetBoundary(int dim, int tag);

  // Get the upward and downward adjacencies of the given entity and return a
  // pair containing a list of tags corresponding to the upward adjacent
  // entities and the downwrad adjacent entities.
  static std::pair<std::vector<int>, std::vector<int>> GetAdjacencies(int dim,
                                                                      int tag);

  // Get the list of lines in the mesh and return a map from the line tags to
  // the node tags adjacent to each line.
  static std::unordered_map<Tag, std::pair<Tag, Tag>> GetLines(int tag = -1);

  // Get the list of triangular faces in the mesh and return a map from the face
  // tags to the node tags belonging to each face.
  static std::unordered_map<Tag, std::tuple<Tag, Tag, Tag>> GetFaces(
      int tag = -1);

  // Get the list of tetrahedral volumes in the mesh and return a map from the
  // tetrahedra tags to the node tags belonging to each tetrahedron.
  static std::unordered_map<Tag, std::tuple<Tag, Tag, Tag, Tag>> GetTetrahedra(
      int tag = -1);

  // Get the list of physical groups in the mesh and return a list of pairs
  // containing the dimension and tag of each physical group.
  static std::vector<DimTag> GetPhysicalGroups(int dim = -1);

  // Get the list of entities making up the given physical group and return the
  // list of entity tags.
  static std::vector<int> GetEntitiesForPhysicalGroups(int dim, int tag);

  // Get the list of physical groups to which the given entity belons and return
  // the list of physical group tags.
  static std::vector<int> GetPhysicalGroupsForEntity(int dim, int tag);
};

}  // namespace gmsh
