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
  Gl* gl;
  std::vector<Volume *> volumes;
  Eigen::Vector3i volumeAtlasSize;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_MANAGER_H_
