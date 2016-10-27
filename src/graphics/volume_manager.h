#ifndef SRC_GRAPHICS_VOLUME_MANAGER_H_

#define SRC_GRAPHICS_VOLUME_MANAGER_H_

#include <Eigen/Core>
#include <map>
#include <vector>
#include "./volume_data.h"
#include "./render_data.h"

namespace Graphics
{

class Gl;
class Volume;

/**
 * \brief Creates and manages a texture atlas for volumes
 *
 * Volume%s are added via #addVolume.
 */
class VolumeManager
{
 public:
  VolumeManager() = default;

  void updateStorage(Gl *gl);
  int addVolume(Volume *volume, Gl *gl);
  void removeVolume(int id);
  std::vector<VolumeData> getBufferData(const RenderData &renderData);
  Eigen::Vector3i getVolumeAtlasSize() const;

  void add3dTexture(int volumeId, Eigen::Vector3i size, float *data,
                    int voxelZOffset);
  void bind(int textureUnit);

 private:
  int nextVolumeId = 1;
  unsigned int texture = 0;
  Gl *gl = nullptr;
  std::map<int, Volume *> volumes;
  std::map<int, Eigen::Matrix4f> objectToDatasetMatrices;
  Eigen::Vector3i volumeAtlasSize = Eigen::Vector3i::Zero();

  const int zPadding = 2;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_MANAGER_H_
