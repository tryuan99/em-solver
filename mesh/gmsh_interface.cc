#include "mesh/gmsh_interface.h"

#include <gmsh.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gmsh {

GmshInterface::GmshInterface() { gmsh::initialize(); }

GmshInterface::~GmshInterface() { gmsh::finalize(); }

void GmshInterface::WriteMeshFile(const std::string& mesh_file) {
  gmsh::write(mesh_file);
}

void GmshInterface::LaunchGui() { gmsh::fltk::run(); }

std::vector<Tag> GmshInterface::GetNodes(const int tag, const int dim,
                                         const NodeType node_type) {
  // Get all node tags.
  std::vector<Tag> node_tags;
  std::vector<double> node_coordinates;
  std::vector<double> node_parametric_coordinates;
  gmsh::model::mesh::getNodes(
      node_tags, node_coordinates, node_parametric_coordinates, dim, tag,
      /*includeBoundary=*/true, /*returnParametricCoord=*/false);

  // Get internal node tags.
  std::vector<Tag> internal_node_tags;
  std::vector<double> internal_node_coordinates;
  std::vector<double> internal_node_parametric_coordinates;
  gmsh::model::mesh::getNodes(internal_node_tags, internal_node_coordinates,
                              internal_node_parametric_coordinates, dim, tag,
                              /*includeBoundary=*/false,
                              /*returnParametricCoord=*/false);

  switch (node_type) {
    case NodeType::kAll: {
      return node_tags;
    }
    case NodeType::kInternal: {
      return internal_node_tags;
    }
    case NodeType::kBoundary: {
      // Get boundary node tags.
      std::unordered_set<Tag> internal_node_tags_set(
          internal_node_tags.cbegin(), internal_node_tags.cend());
      std::vector<Tag> boundary_node_tags;
      std::copy_if(node_tags.cbegin(), node_tags.cend(),
                   std::back_inserter(boundary_node_tags), [&](const Tag tag) {
                     return !internal_node_tags_set.contains(tag);
                   });
      return boundary_node_tags;
    }
    default: {
      throw std::invalid_argument("Invalid node type.");
    }
  }
}

std::unordered_map<Tag, Coordinates> GmshInterface::GetNodeCoordinates(
    const int tag, const int dim) {
  // Get all node tags and coordinates.
  std::vector<Tag> node_tags;
  std::vector<double> node_coordinates;
  std::vector<double> node_parametric_coordinates;
  gmsh::model::mesh::getNodes(
      node_tags, node_coordinates, node_parametric_coordinates, dim, tag,
      /*includeBoundary=*/true, /*returnParametricCoord=*/false);

  // Insert the node tags and coordinates into a map.
  std::unordered_map<Tag, Coordinates> node_tag_to_coordinates;
  for (uint32_t i = 0; i < node_tags.size(); ++i) {
    node_tag_to_coordinates.try_emplace(
        node_tags[i], std::array<double, 3>{node_coordinates[3 * i],
                                            node_coordinates[3 * i + 1],
                                            node_coordinates[3 * i + 2]});
  }
  return node_tag_to_coordinates;
}

std::vector<DimTag> GetBoundary(const int dim, const int tag) {
  std::vector<DimTag> input_dim_tags{{dim, tag}};
  std::vector<DimTag> output_dim_tags;
  gmsh::model::getBoundary(input_dim_tags, output_dim_tags, /*combined=*/true,
                           /*oriented=*/false, /*recursive=*/false);
  return output_dim_tags;
}

std::pair<std::vector<int>, std::vector<int>> GmshInterface::GetAdjacencies(
    const int dim, const int tag) {
  std::vector<int> upward_adjacencies;
  std::vector<int> downward_adjacencies;
  gmsh::model::getAdjacencies(dim, tag, upward_adjacencies,
                              downward_adjacencies);
  return std::make_pair(std::move(upward_adjacencies),
                        std::move(downward_adjacencies));
}

std::unordered_map<Tag, std::array<Tag, 2>> GmshInterface::GetLines(
    const int tag) {
  std::vector<Tag> line_tags;
  std::vector<Tag> node_tags;
  gmsh::model::mesh::getElementsByType(
      static_cast<int>(ElementType::kTwoNodeLine), line_tags, node_tags, tag);

  // Insert the line and node tags into a map.
  std::unordered_map<Tag, std::array<Tag, 2>> lines;
  for (uint32_t i = 0; i < line_tags.size(); ++i) {
    lines.try_emplace(
        line_tags[i],
        std::array<gmsh::Tag, 2>{node_tags[2 * i], node_tags[2 * i + 1]});
  }
  return lines;
}

std::unordered_map<Tag, std::array<Tag, 3>> GmshInterface::GetFaces(
    const int tag) {
  std::vector<Tag> face_tags;
  std::vector<Tag> node_tags;
  gmsh::model::mesh::getElementsByType(
      static_cast<int>(ElementType::kThreeNodeTriangle), face_tags, node_tags,
      tag);

  // Insert the face and node tags into a map.
  std::unordered_map<Tag, std::array<Tag, 3>> faces;
  for (uint32_t i = 0; i < face_tags.size(); ++i) {
    faces.try_emplace(face_tags[i], std::array<gmsh::Tag, 3>{
                                        node_tags[3 * i], node_tags[3 * i + 1],
                                        node_tags[3 * i + 2]});
  }
  return faces;
}

std::unordered_map<Tag, std::array<Tag, 4>> GmshInterface::GetTetrahedra(
    const int tag) {
  std::vector<Tag> volume_tags;
  std::vector<Tag> node_tags;
  gmsh::model::mesh::getElementsByType(
      static_cast<int>(ElementType::kFourNodeTetrahedron), volume_tags,
      node_tags, tag);

  // Insert the volume and node tags into a map.
  std::unordered_map<Tag, std::array<Tag, 4>> volumes;
  for (uint32_t i = 0; i < volume_tags.size(); ++i) {
    volumes.try_emplace(
        volume_tags[i],
        std::array<gmsh::Tag, 4>{node_tags[4 * i], node_tags[4 * i + 1],
                                 node_tags[4 * i + 2], node_tags[4 * i + 3]});
  }
  return volumes;
}

std::vector<DimTag> GmshInterface::GetPhysicalGroups(const int dim) {
  std::vector<DimTag> dim_tags;
  gmsh::model::getPhysicalGroups(dim_tags, dim);
  return dim_tags;
}

std::vector<int> GmshInterface::GetEntitiesForPhysicalGroups(const int dim,
                                                             const int tag) {
  std::vector<int> entity_tags;
  gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entity_tags);
  return entity_tags;
}

std::vector<int> GmshInterface::GetPhysicalGroupsForEntity(const int dim,
                                                           const int tag) {
  std::vector<int> physical_group_tags;
  gmsh::model::getPhysicalGroupsForEntity(dim, tag, physical_group_tags);
  return physical_group_tags;
}

}  // namespace gmsh
