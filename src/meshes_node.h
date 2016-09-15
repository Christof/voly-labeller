#ifndef SRC_MESHES_NODE_H_

#define SRC_MESHES_NODE_H_

#include <Eigen/Core>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <QDebug>
#include <memory>
#include <vector>
#include <string>
#include "./node.h"
#include "./math/obb.h"
#include "./graphics/mesh.h"
#include "./graphics/gl.h"

/**
 * \brief Node which renders a Mesh
 */
class MeshesNode : public Node
{
 public:
  /**
   * \brief Construct meshes from the given asset filename
   */
  MeshesNode(std::string assetFilename,
           Eigen::Matrix4f transformation);
  virtual ~MeshesNode();

  virtual void render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData);

  Eigen::Matrix4f getTransformation();
  void setTransformation(Eigen::Matrix4f transformation);

  template <class Archive>
  void save_construct_data(Archive &ar, const MeshesNode *meshesNode,
                           const unsigned int file_version) const
  {
    qDebug() << "In save_construct_data method";
    assert(meshesNode != nullptr && "MeshesNode was null");

    ar << BOOST_SERIALIZATION_NVP(assetFilename);
    ar << BOOST_SERIALIZATION_NVP(transformation);
  };

 private:
  friend class boost::serialization::access;

  void loadMeshes();
  void createObbVisualization();

  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
    boost::serialization::void_cast_register<MeshesNode, Node>(
        static_cast<MeshesNode *>(NULL), static_cast<Node *>(NULL));
  };

  std::string assetFilename;
  std::vector<std::shared_ptr<Graphics::Mesh>> meshes;
  std::vector<Eigen::Matrix4f> transformations;
  Eigen::Matrix4f transformation;
};

namespace boost
{
namespace serialization
{

template <class Archive>
inline void save_construct_data(Archive &ar, const MeshesNode *meshesNode,
                                const unsigned int file_version)
{
  qDebug() << "In save_construct_data function";
  assert(meshesNode != nullptr && "MeshesNode was null");

  meshesNode->save_construct_data(ar, meshesNode, file_version);
}

template <class Archive>
inline void load_construct_data(Archive &ar, MeshesNode *t,
                                const unsigned int version)
{
  std::string assetFilename;
  Eigen::Matrix4f transformation;
  ar >> BOOST_SERIALIZATION_NVP(assetFilename);
  ar >> BOOST_SERIALIZATION_NVP(transformation);

  ::new (t) MeshesNode(assetFilename, transformation);
}
}  // namespace serialization
}  // namespace boost


#endif  // SRC_MESHES_NODE_H_
