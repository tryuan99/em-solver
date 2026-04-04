#include "solver/node_index_lookup.h"

#include <algorithm>
#include <stdexcept>

namespace solver {

NodeIndexLookup::NodeIndexLookup(const std::vector<gmsh::Tag>& node_tags)
    : node_tags_(node_tags) {
  std::sort(node_tags_.begin(), node_tags_.end());
  node_tags_.erase(std::unique(node_tags_.begin(), node_tags_.end()),
                   node_tags_.end());

  for (std::size_t index = 0; index < node_tags_.size(); ++index) {
    node_tag_to_index_.emplace(node_tags_[index], index);
  }
}

std::size_t NodeIndexLookup::index_from_tag(const gmsh::Tag tag) const {
  const auto it = node_tag_to_index_.find(tag);
  if (it == node_tag_to_index_.cend()) {
    throw std::out_of_range("Node tag cannot be found.");
  }
  return it->second;
}

gmsh::Tag NodeIndexLookup::tag_from_index(const std::size_t index) const {
  if (index >= node_tags_.size()) {
    throw std::out_of_range("Node index is out of range.");
  }
  return node_tags_[index];
}

}  // namespace solver
