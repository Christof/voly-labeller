#ifndef SRC_MESH_NODE_H_

#define SRC_MESH_NODE_H_

#include <Eigen/Core>
#include <memory>
#include "./node.h"

class Mesh;
/**
 * \brief
 *
 *
 */
class MeshNode : public Node
{
 public:
  MeshNode(std::shared_ptr<Mesh> mesh, Eigen::Matrix4f transformation);
  virtual ~MeshNode();

  void render(const RenderData &renderData);

  Eigen::Matrix4f getTransformation();
  void setTransformation(Eigen::Matrix4f transformation);

 private:
  std::shared_ptr<Mesh> mesh;
  Eigen::Matrix4f transformation;
};

#endif  // SRC_MESH_NODE_H_
