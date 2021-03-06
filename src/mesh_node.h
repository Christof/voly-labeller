#ifndef SRC_MESH_NODE_H_

#define SRC_MESH_NODE_H_

#include <Eigen/Core>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <QDebug>
#include <memory>
#include <string>
#include "./node.h"
#include "./math/obb.h"
#include "./importer.h"
#include "./graphics/mesh.h"
#include "./graphics/gl.h"

/**
 * \brief Node which renders a Mesh
 */
class MeshNode : public Node
{
 public:
  /**
   * \brief Construct the mesh from the given asset filename and mesh index in
   * that asset file
   */
  MeshNode(std::string assetFilename, int meshIndex,
           std::shared_ptr<Graphics::Mesh> mesh,
           Eigen::Matrix4f transformation);
  virtual ~MeshNode();

  virtual void render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData);

  Eigen::Matrix4f getTransformation();
  void setTransformation(Eigen::Matrix4f transformation);

  template <class Archive>
  void save_construct_data(Archive &ar, const MeshNode *meshNode,
                           const unsigned int file_version) const
  {
    qDebug() << "In save_construct_data method";
    assert(meshNode != nullptr && "MeshNode was null");

    ar << BOOST_SERIALIZATION_NVP(assetFilename);
    ar << BOOST_SERIALIZATION_NVP(meshIndex);
    ar << BOOST_SERIALIZATION_NVP(transformation);
  };

 private:
  friend class boost::serialization::access;

  void createObbVisualization();

  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
    boost::serialization::void_cast_register<MeshNode, Node>(
        static_cast<MeshNode *>(NULL), static_cast<Node *>(NULL));
  };

  std::string assetFilename;
  int meshIndex;
  std::shared_ptr<Graphics::Mesh> mesh;
  Eigen::Matrix4f transformation;
};

namespace boost
{
namespace serialization
{

template <class Archive>
inline void save_construct_data(Archive &ar, const MeshNode *meshNode,
                                const unsigned int file_version)
{
  qDebug() << "In save_construct_data function";
  assert(meshNode != nullptr && "MeshNode was null");

  meshNode->save_construct_data(ar, meshNode, file_version);
}

template <class Archive>
inline void load_construct_data(Archive &ar, MeshNode *t,
                                const unsigned int version)
{
  std::string assetFilename;
  int meshIndex;
  Eigen::Matrix4f transformation;
  ar >> BOOST_SERIALIZATION_NVP(assetFilename);
  ar >> BOOST_SERIALIZATION_NVP(meshIndex);
  ar >> BOOST_SERIALIZATION_NVP(transformation);

  Importer importer;

  ::new (t) MeshNode(assetFilename, meshIndex,
                     importer.import(assetFilename, meshIndex), transformation);
}
}  // namespace serialization
}  // namespace boost
#endif  // SRC_MESH_NODE_H_
