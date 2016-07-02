#ifndef SRC_CAMERA_NODE_H_

#define SRC_CAMERA_NODE_H_

#include <Eigen/Core>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <memory>
#include <vector>
#include <string>
#include "./node.h"
#include "./camera.h"

namespace Graphics
{
class Gl;
class Managers;
}

class CameraPosition
{
 public:
  CameraPosition() = default;
  CameraPosition(std::string name, Eigen::Matrix4f viewMatrix)
    : name(name), viewMatrix(viewMatrix)
  {
  }

  std::string name;
  Eigen::Matrix4f viewMatrix;
};

/**
 * \brief Node for a Camera
 *
 * This is used so that the camera settings are saved in the scene.
 */
class CameraNode : public Node
{
 public:
  CameraNode();
  CameraNode(std::shared_ptr<Camera> camera,
             std::vector<CameraPosition> cameraPositions);

  std::shared_ptr<Camera> getCamera();

  virtual void render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData);

  template <class Archive> void save_construct_data(Archive &ar) const
  {
    Eigen::Matrix4f viewMatrix = camera->getViewMatrix();
    Eigen::Matrix4f projectionMatrix = camera->getProjectionMatrix();
    Eigen::Vector3f origin = camera->getOrigin();

    ar << BOOST_SERIALIZATION_NVP(viewMatrix);
    ar << BOOST_SERIALIZATION_NVP(projectionMatrix);
    ar << BOOST_SERIALIZATION_NVP(origin);
    ar << BOOST_SERIALIZATION_NVP(cameraPositions);
  }

  std::vector<CameraPosition> cameraPositions;

  void setOnCameraPositionsChanged(
      std::function<void(std::vector<CameraPosition>)> onChanged);

  void saveCameraPosition(std::string name, Eigen::Matrix4f viewMatrix);
  void removeCameraPosition(int index);
  void changeCameraPositionName(int index, std::string newName);

 private:
  std::shared_ptr<Camera> camera;
  std::function<void(std::vector<CameraPosition>)> onCameraPositionsChanged;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
    boost::serialization::void_cast_register<CameraNode, Node>(
        static_cast<CameraNode *>(NULL), static_cast<Node *>(NULL));
  }
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
  std::vector<CameraPosition> cameraPositions;
  if (version > 0)
    ar >> BOOST_SERIALIZATION_NVP(cameraPositions);

  auto camera = std::make_shared<Camera>(viewMatrix, projectionMatrix, origin);
  ::new (t) CameraNode(camera, cameraPositions);
}

template <class Archive>
void save(Archive &ar, const CameraPosition &t, unsigned int version)
{
  auto name = t.name;
  ar << BOOST_SERIALIZATION_NVP(name);
  auto viewMatrix = t.viewMatrix;
  ar << BOOST_SERIALIZATION_NVP(viewMatrix);
}
template <class Archive>
void load(Archive &ar, CameraPosition &t, unsigned int version)
{
  std::string name;
  ar >> BOOST_SERIALIZATION_NVP(name);
  Eigen::Matrix4f viewMatrix;
  ar >> BOOST_SERIALIZATION_NVP(viewMatrix);

  t.name = name;
  t.viewMatrix = viewMatrix;
}

}  // namespace serialization
}  // namespace boost

BOOST_SERIALIZATION_SPLIT_FREE(CameraPosition)
BOOST_CLASS_VERSION(CameraNode, 1)

#endif  // SRC_CAMERA_NODE_H_
