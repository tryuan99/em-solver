// The neighbor lookup class facilitates looking up the neighboring nodes along
// the adjacent entities, i.e., faces or tetrahera, of a given node.
//
// The neighbors are stored in a map from the node tag to a list of length equal
// to the number of adjacent entities. This list contains arrays of node tags,
// each of length 2 in the case of triangles or length 3 in the case of
// tetrahedra.

#pragma once

#include <cstdlib>
#include <list>
#include <unordered_map>
#include <utility>

#include "mesh/gmsh_interface.h"

namespace solver {

// Neighbor lookup.
template <std::size_t Dimension>
class NeighborLookup {
 public:
  using T = std::array<gmsh::Tag, Dimension>;
  using U = std::array<gmsh::Tag, Dimension - 1>;

  NeighborLookup(const std::unordered_map<gmsh::Tag, T>& element_tag_to_nodes)
      : neighbors_(InitNeighbors(element_tag_to_nodes)) {}

  // Get the number of adjancent entities to the given node.
  std::size_t num_adjacent_entities(const int tag) const {
    return neighbors_[tag].size();
  }

  // Get the neighbors of the given node.
  const std::list<U>& neighbors(const int tag) const { return neighbors_[tag]; }

 private:
  // Initialize the neighbors.
  static std::unordered_map<gmsh::Tag, std::list<U>> InitNeighbors(
      const std::unordered_map<gmsh::Tag, T>& element_tag_to_nodes);

  // Map from the node tag to the list of node tag to the list of node tags of
  // the adjacent entities.
  std::unordered_map<gmsh::Tag, std::list<U>> neighbors_;
};

// Type definitions.
using NeighborLookup2D = NeighborLookup<2>;
using NeighborLookup3D = NeighborLookup<3>;

}  // namespace solver
