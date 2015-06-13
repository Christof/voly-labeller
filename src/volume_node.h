#ifndef SRC_VOLUME_NODE_H_

#define SRC_VOLUME_NODE_H_

#include <string>
#include <memory>
#include "./node.h"
#include "./gl.h"
#include "./math/obb.h"

struct RenderData;
class VolumeReader;
class Quad;
class Cube;

/**
 * \brief
 *
 *
 */
class VolumeNode : public Node
{
 public:
  explicit VolumeNode(std::string filename);
  virtual ~VolumeNode();

  void render(Gl *gl, RenderData renderData);

  template <class Archive> void save_construct_data(Archive &ar) const
  {
    ar << BOOST_SERIALIZATION_NVP(filename);
  };

  Eigen::Matrix4f getTransformation();
  virtual std::shared_ptr<Math::Obb> getObb();
 private:
  std::string filename;
  std::unique_ptr<VolumeReader> volumeReader;
  std::unique_ptr<Quad> quad;
  std::unique_ptr<Cube> cube;
  std::shared_ptr<Math::Obb> obb;
  GLuint texture = 0;

  void initializeTexture(Gl *gl);

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
    boost::serialization::void_cast_register<VolumeNode, Node>(
        static_cast<VolumeNode *>(NULL), static_cast<Node *>(NULL));
  };
};

namespace boost
{
namespace serialization
{

template <class Archive>
inline void save_construct_data(Archive &ar, const VolumeNode *volumeNode,
                                const unsigned int file_version)
{
  volumeNode->save_construct_data(ar);
}

template <class Archive>
inline void load_construct_data(Archive &ar, VolumeNode *t,
                                const unsigned int version)
{
  std::string filename;
  ar >> BOOST_SERIALIZATION_NVP(filename);

  ::new (t) VolumeNode(filename);
}
}  // namespace serialization
}  // namespace boost

#endif  // SRC_VOLUME_NODE_H_
