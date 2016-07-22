#if _WIN32
#pragma warning(disable : 4522)
#endif

#include "./volume_manager.h"
#include <QLoggingCategory>
#include <Eigen/Geometry>
#include <algorithm>
#include <vector>
#include "../eigen_qdebug.h"
#include "./gl.h"
#include "./volume.h"

namespace Graphics
{

QLoggingCategory vmChan("Graphics.VolumeManager");

void VolumeManager::updateStorage(Gl *gl)
{
  qCInfo(vmChan) << "initialize: current texture:" << texture;

  if (texture)
  {
    qCInfo(vmChan) << "deleting old texture";
    gl->glDeleteTextures(1, &texture);
    volumeAtlasSize = Eigen::Vector3i::Zero();
  }

  this->gl = gl;

  if (volumes.size() == 0)
    return;

  for (auto iterator = volumes.cbegin(); iterator != volumes.cend();)
  {
    auto volumeSize = iterator->second->getDataSize();
    qCInfo(vmChan) << "adding volume with size:" << volumeSize.x()
                   << volumeSize.y() << volumeSize.z();
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
  volume->initialize(nextVolumeId, this);

  updateStorage(gl);

  return nextVolumeId++;
}

void VolumeManager::removeVolume(int id)
{
  volumes.erase(volumes.find(id));
}

std::vector<VolumeData>
VolumeManager::getBufferData(const RenderData &renderData)
{
  std::vector<VolumeData> data;
  for (int i = 1; i < nextVolumeId; ++i)
  {
    VolumeData volumeData;
    if (volumes.count(i))
    {
      volumeData = volumes[i]->getVolumeData(renderData);
      volumeData.objectToDatasetMatrix =
          objectToDatasetMatrices[volumeData.volumeId];
    }

    data.push_back(volumeData);
  }

  return data;
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

  gl->glTexParameterf(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl->glTexParameterf(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl->glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  gl->glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  gl->glTexParameteri(textureTarget, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
}

}  // namespace Graphics
