#ifndef SRC_MESH_NODE_H_

#define SRC_MESH_NODE_H_

#include <Eigen/Core>
#include <memory>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/access.hpp>
#include "./node.h"
#include "./importer.h"

class Mesh;
/**
 * \brief
 *
 *
 */
class MeshNode : public Node
{
 public:
  MeshNode(std::string assetFilename, int meshIndex, std::shared_ptr<Mesh> mesh,
           Eigen::Matrix4f transformation);
  virtual ~MeshNode();

  void render(const RenderData &renderData);

  Eigen::Matrix4f getTransformation();
  void setTransformation(Eigen::Matrix4f transformation);

 private:
  friend class boost::serialization::access;
  template <class Archive> void save(Archive &ar, unsigned int version) const
  {
    ar << BOOST_SERIALIZATION_NVP(assetFilename);
    ar << BOOST_SERIALIZATION_NVP(meshIndex);
    ar << BOOST_SERIALIZATION_NVP(transformation);
  };
  template <class Archive>
  void load(Archive &ar, unsigned int version)
  {
  };
  std::string assetFilename;
  int meshIndex;
  std::shared_ptr<Mesh> mesh;
  Eigen::Matrix4f transformation;
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

// BOOST_SERIALIZATION_SPLIT_FREE(MeshNode)

namespace boost
{
namespace serialization
{
template <class Archive>
inline void load_construct_data(Archive &ar, MeshNode *t,
                                const unsigned int version)
{
  // retrieve data from archive required to construct new instance
  std::string assetFilename;
  int meshIndex;
  Eigen::Matrix4f transformation;
  ar >> assetFilename;
  ar >> meshIndex;
  ar >> transformation;

  Importer importer(Gl::instance);
  // invoke inplace constructor to initialize instance of my_class
  ::new (t) MeshNode(assetFilename, meshIndex,
                     importer.import(assetFilename, meshIndex), transformation);
}
}  // namespace serialization
}  // namespace boost
#endif  // SRC_MESH_NODE_H_
