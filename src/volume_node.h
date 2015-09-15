#ifndef SRC_VOLUME_NODE_H_

#define SRC_VOLUME_NODE_H_

#include <string>
#include <memory>
#include "./node.h"
#include "./graphics/gl.h"
#include "./graphics/cube.h"
#include "./graphics/object_data.h"
#include "./graphics/volume.h"
#include "./graphics/transfer_function_manager.h"

struct RenderData;
class VolumeReader;

/**
 * \brief Node which renders a volume
 *
 */
class VolumeNode : public Node, public Graphics::Volume
{
 public:
  explicit VolumeNode(std::string filename);
  virtual ~VolumeNode();

  void render(Graphics::Gl *gl, RenderData renderData);

  virtual Graphics::VolumeData getVolumeData();
  virtual float* getData();
  virtual Eigen::Vector3i getDataSize();

  template <class Archive> void save_construct_data(Archive &ar) const
  {
    ar << BOOST_SERIALIZATION_NVP(filename);
  };

  Eigen::Matrix4f getTransformation();

 private:
  std::string filename;
  std::unique_ptr<VolumeReader> volumeReader;
  std::unique_ptr<Graphics::Cube> cube;
  GLuint texture = 0;
  Graphics::ObjectData cubeData;
  int volumeId;
  int textureId = 0;
  int transferFunctionRow = -1;

  static std::unique_ptr<Graphics::TransferFunctionManager>
      transferFunctionManager;

  void initialize(Graphics::Gl *gl);

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
