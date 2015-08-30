#ifndef SRC_GRAPHICS_VOLUME_MANAGER_H_

#define SRC_GRAPHICS_VOLUME_MANAGER_H_

#include <vector>
#include <Eigen/Core>
#include "./volume.h"
#include "./object_data.h"

namespace Graphics
{


/**
 * \brief
 *
 *
 */
class VolumeManager
{
 public:
  VolumeManager() = default;

  int addVolume(Volume* volume);
  void fillCustomBuffer(ObjectData &objectData);

  static VolumeManager* instance;
 private:

  int nextVolumeId = 1;

  std::vector<Volume*> volumes;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_MANAGER_H_
