#include "solver/neighbor_lookup.h"

#include <algorithm>
#include <cstdlib>
#include <list>
#include <unordered_map>

#include "mesh/gmsh_interface.h"

namespace solver {

template <std::size_t Dimension>
std::unordered_map<gmsh::Tag, std::list<typename NeighborLookup<Dimension>::U>>
NeighborLookup<Dimension>::InitNeighbors(
    const std::unordered_map<gmsh::Tag, T>& element_tag_to_nodes) {
  std::unordered_map<gmsh::Tag, std::list<U>> neighbors;
  for (const auto& [_, node_tags] : element_tag_to_nodes) {
    for (const auto node_tag : node_tags) {
      auto& neighbor_tags = neighbors[node_tag].emplace_back();
      std::copy_if(node_tags.cbegin(), node_tags.cend(), neighbor_tags.begin(),
                   [&](const gmsh::Tag tag) { return tag != node_tag; });
    }
  }
  return neighbors;
}

// Explicit instantiations of the template.
template class NeighborLookup<2>;
template class NeighborLookup<3>;

}  // namespace solver
