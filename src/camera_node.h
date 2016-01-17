#ifndef SRC_CAMERA_NODE_H_

#define SRC_CAMERA_NODE_H_

#include <Eigen/Core>
#include "./node.h"
#include "./graphics/gl.h"
#include "./graphics/managers.h"
#include "./camera.h"

/**
 * \brief
 *
 *
 */
class CameraNode : public Node
{
 public:
  CameraNode() = default;
  CameraNode(Camera camera);

  Camera &getCamera();

  virtual void render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData);

  template <class Archive> void save_construct_data(Archive &ar) const
  {
    Eigen::Matrix4f viewMatrix = camera.getViewMatrix();
    Eigen::Matrix4f projectionMatrix = camera.getProjectionMatrix();
    Eigen::Vector3f origin = camera.getOrigin();

    ar << BOOST_SERIALIZATION_NVP(viewMatrix);
    ar << BOOST_SERIALIZATION_NVP(projectionMatrix);
    ar << BOOST_SERIALIZATION_NVP(origin);
  };

 private:
  Camera camera;
};

namespace boost
{
namespace serialization
{

template <class Archive>
inline void save_construct_data(Archive &ar, const CameraNode *cameraNode,
                                const unsigned int file_version)
{
  cameraNode->save_construct_data(ar);
}

template <class Archive>
inline void load_construct_data(Archive &ar, CameraNode *t,
                                const unsigned int version)
{
  Eigen::Matrix4f viewMatrix;
  ar >> BOOST_SERIALIZATION_NVP(viewMatrix);
  Eigen::Matrix4f projectionMatrix;
  ar >> BOOST_SERIALIZATION_NVP(projectionMatrix);
  Eigen::Vector3f origin;
  ar >> BOOST_SERIALIZATION_NVP(origin);

  ::new (t) CameraNode(Camera(viewMatrix, projectionMatrix, origin));
}
}  // namespace serialization
}  // namespace boost

#endif  // SRC_CAMERA_NODE_H_
