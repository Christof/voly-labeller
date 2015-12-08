#include "./volume_manager.h"
#include <QLoggingCategory>
#include <Eigen/Geometry>
#include <algorithm>
#include <vector>
#include "../eigen_qdebug.h"
#include "./gl.h"

namespace Graphics
{

QLoggingCategory vmChan("Graphics.VolumeManager");

void VolumeManager::updateStorage(Gl *gl)
{
  if (texture)
    gl->glDeleteTextures(1, &texture);

  qCInfo(vmChan) << "initialize";
  this->gl = gl;

  if (volumes.size() == 0)
    return;

  for (auto iterator = volumes.cbegin(); iterator != volumes.cend();)
  {
    auto volumeSize = iterator->second->getDataSize();
    volumeAtlasSize.z() += volumeSize.z();
    volumeAtlasSize.x() = std::max(volumeSize.x(), volumeAtlasSize.x());
    volumeAtlasSize.y() = std::max(volumeSize.y(), volumeAtlasSize.y());

    ++iterator;
    if (iterator != volumes.cend())
      volumeAtlasSize.z() += zPadding;
  }
  qCInfo(vmChan) << "volumeAtlasSize" << volumeAtlasSize;

  gl->glGenTextures(1, &texture);
  gl->glBindTexture(GL_TEXTURE_3D, texture);
  gl->glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32F, volumeAtlasSize.x(),
                     volumeAtlasSize.y(), volumeAtlasSize.z());

  float zero = 0.0f;
  gl->glClearTexImage(texture, 0, GL_RED, GL_FLOAT, &zero);

  int voxelZOffset = 0;
  for (auto volumePair : volumes)
  {
    auto volume = volumePair.second;
    add3dTexture(volumePair.first, volume->getDataSize(), volume->getData(),
                 voxelZOffset);
    voxelZOffset += volume->getDataSize().z() + zPadding;
  }
}

int VolumeManager::addVolume(Volume *volume, Gl *gl)
{
  qCInfo(vmChan) << "addVolume";

  volumes[nextVolumeId] = volume;
  int volumeToDelete = nextVolumeId;
  volume->setRemoveFromVolumesFunction([volumeToDelete, this]()
                                       {
                                         removeVolume(volumeToDelete);
                                       });

  updateStorage(gl);

  return nextVolumeId++;
}

void VolumeManager::removeVolume(int id)
{
  volumes.erase(volumes.find(id));
}

void VolumeManager::fillCustomBuffer(ObjectData &objectData)
{
  int size = sizeof(VolumeData) * (nextVolumeId - 1);
  std::vector<VolumeData> data;
  for (int i = 1; i < nextVolumeId; ++i)
  {
    VolumeData volumeData;
    if (volumes.count(i))
    {
      volumeData = volumes[i]->getVolumeData();
      volumeData.objectToDatasetMatrix =
          objectToDatasetMatrices[volumeData.volumeId];
    }

    data.push_back(volumeData);
  }

  objectData.setCustomBuffer(size, [data, size](void *insertionPoint)
                             {
    std::memcpy(insertionPoint, data.data(), size);
  });
}

Eigen::Vector3i VolumeManager::getVolumeAtlasSize() const
{
  return volumeAtlasSize;
}

void VolumeManager::add3dTexture(int volumeId, Eigen::Vector3i size,
                                 float *data, int voxelZOffset)
{
  auto textureTarget = GL_TEXTURE_3D;

  gl->glBindTexture(textureTarget, texture);
  gl->glTexSubImage3D(textureTarget, 0, 0, 0, voxelZOffset, size.x(), size.y(),
                      size.z(), GL_RED, GL_FLOAT, data);

  Eigen::Vector3f scaling =
      size.cast<float>().cwiseQuotient(volumeAtlasSize.cast<float>());
  Eigen::Affine3f transformation(
      Eigen::Translation3f(Eigen::Vector3f(
          0, 0, voxelZOffset / static_cast<float>(volumeAtlasSize.z()))) *
      Eigen::Scaling(scaling));

  Eigen::Matrix4f matrix = transformation.matrix();
  qCInfo(vmChan) << "transformation for" << volumeId << matrix << "size"
                 << size;
  objectToDatasetMatrices[volumeId] = matrix;

  glTexParameterf(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
}

}  // namespace Graphics
