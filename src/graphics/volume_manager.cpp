#include "./volume_manager.h"

namespace Graphics
{

VolumeManager *VolumeManager::instance = new VolumeManager();

int VolumeManager::addVolume(Volume *volume)
{
  volumes.push_back(volume);
  return nextVolumeId++;
}
void VolumeManager::fillCustomBuffer(ObjectData &objectData)
{
  int size = sizeof(VolumeData) * volumes.size();
  std::vector<VolumeData> data;
  for (auto volume : volumes)
    data.push_back(volume->getVolumeData());

  objectData.setCustomBuffer(size, [data, size](void *insertionPoint)
                             {
    std::memcpy(insertionPoint, data.data(), size);
  });
}

}  // namespace Graphics
