#include "./volume_manager.h"
#include <QLoggingCategory>
#include <Eigen/Geometry>
#include "../eigen_qdebug.h"
#include "./gl.h"

namespace Graphics
{

QLoggingCategory vmChan("Graphics.VolumeManager");

VolumeManager *VolumeManager::instance = new VolumeManager();

void VolumeManager::initialize(Gl *gl)
{
  qCInfo(vmChan) << "initialize";
  this->gl = gl;

  gl->glGenTextures(1, &texture);
  gl->glBindTexture(GL_TEXTURE_3D, texture);
  gl->glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, volumeAtlasSize.x(),
                   volumeAtlasSize.y(), volumeAtlasSize.z(), 0, GL_RED,
                   GL_FLOAT, nullptr);
}

int VolumeManager::addVolume(Volume *volume)
{
  qCInfo(vmChan) << "addVolume";

  volumes.push_back(volume);
  return nextVolumeId++;
}

void VolumeManager::fillCustomBuffer(ObjectData &objectData)
{
  int size = sizeof(VolumeData) * volumes.size();
  std::vector<VolumeData> data;
  for (auto volume : volumes)
  {
    auto volumeData = volume->getVolumeData();
    volumeData.objectToDatasetMatrix =
        objectToDatasetMatrices[1];
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

unsigned int VolumeManager::add3dTexture(Eigen::Vector3i size, float *data)
{
  auto textureTarget = GL_TEXTURE_3D;

  gl->glBindTexture(textureTarget, texture);
  Eigen::Vector3i offset(0, 0, 0);
  gl->glTexSubImage3D(textureTarget, 0, offset.x(), offset.y(), offset.z(),
                      size.x(), size.y(), size.z(), GL_RED, GL_FLOAT, data);

  Eigen::Vector3f scaling =
      size.cast<float>().cwiseQuotient(volumeAtlasSize.cast<float>());
  Eigen::Affine3f transformation(Eigen::Scaling(scaling) *
                                 Eigen::Translation3f(offset.cast<float>()));

  qCInfo(vmChan) << "transformation" << transformation.matrix();
  objectToDatasetMatrices[1] = transformation.matrix();

  glTexParameterf(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

  return texture;
}

}  // namespace Graphics
