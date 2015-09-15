#ifndef SRC_GRAPHICS_VOLUME_MANAGER_H_

#define SRC_GRAPHICS_VOLUME_MANAGER_H_

#include <vector>
#include <Eigen/Core>
#include "./volume.h"
#include "./object_data.h"

namespace Graphics
{

class Gl;

/**
 * \brief
 *
 *
 */
class VolumeManager
{
 public:
  VolumeManager() = default;

  void initialize(Gl *gl);
  int addVolume(Volume *volume);
  void fillCustomBuffer(ObjectData &objectData);
  Eigen::Vector3i getVolumeAtlasSize() const;

  unsigned int add3dTexture(Eigen::Vector3i size, float *data);

  static VolumeManager *instance;

 private:
  int nextVolumeId = 1;
  unsigned int texture;
  Gl* gl;
  std::vector<Volume *> volumes;
  std::map<int, Eigen::Matrix4f> objectToDatasetMatrices;
  Eigen::Vector3i volumeAtlasSize = Eigen::Vector3i(1024, 1024, 200);
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_MANAGER_H_
