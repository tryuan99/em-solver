#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "mesh/gmsh_interface.h"

namespace solver {

// Maps arbitrary Gmsh node tags onto dense solver indices.
class NodeIndexLookup {
 public:
  NodeIndexLookup() = default;
  explicit NodeIndexLookup(const std::vector<gmsh::Tag>& node_tags);

  std::size_t size() const { return node_tags_.size(); }

  std::size_t index_from_tag(gmsh::Tag tag) const;

  gmsh::Tag tag_from_index(std::size_t index) const;

 private:
  std::vector<gmsh::Tag> node_tags_;
  std::unordered_map<gmsh::Tag, std::size_t> node_tag_to_index_;
};

}  // namespace solver
