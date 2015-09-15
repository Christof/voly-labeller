#include "./volume_manager.h"
#include "./gl.h"

namespace Graphics
{

VolumeManager *VolumeManager::instance = new VolumeManager();

void VolumeManager::initialize(Gl *gl)
{
  this->gl = gl;
}

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

Eigen::Vector3i VolumeManager::getVolumeAtlasSize() const
{
  return volumeAtlasSize;
}

unsigned int VolumeManager::add3dTexture(Eigen::Vector3i size, float *data)
{
  volumeAtlasSize = size;
  auto textureTarget = GL_TEXTURE_3D;
  unsigned int texture = 0;

  glAssert(gl->glGenTextures(1, &texture));
  glAssert(gl->glBindTexture(textureTarget, texture));
  glAssert(gl->glTexImage3D(textureTarget, 0, GL_R32F, size.x(), size.y(),
                            size.z(), 0, GL_RED, GL_FLOAT, data));

  glTexParameterf(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

  return texture;
}

}  // namespace Graphics
