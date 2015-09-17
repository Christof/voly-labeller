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

  void add3dTexture(int volumeId, Eigen::Vector3i size, float *data,
                    int voxelZOffset);

  static VolumeManager *instance;

 private:
  int nextVolumeId = 1;
  unsigned int texture;
  Gl *gl = nullptr;
  std::vector<Volume *> volumes;
  std::vector<Volume *> volumesToAdd;
  std::map<int, Eigen::Matrix4f> objectToDatasetMatrices;
  Eigen::Vector3i volumeAtlasSize = Eigen::Vector3i::Zero();

  const int zPadding = 2;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_MANAGER_H_
