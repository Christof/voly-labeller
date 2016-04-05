#include "./texture_mapper_manager.h"
#include "./utils/memory.h"
#include "./placement/cuda_texture_mapper.h"
#include "./placement/distance_transform.h"
#include "./placement/occlusion.h"
#include "./placement/apollonius.h"
#include "./utils/image_persister.h"
#include "./graphics/standard_texture_2d.h"
#include "./constraint_buffer_object.h"
#include "./texture_mappers_for_layer.h"

TextureMapperManager::TextureMapperManager(int bufferSize)
  : bufferSize(bufferSize)
{
}

TextureMapperManager::~TextureMapperManager()
{
  for (auto mappers : mappersForLayers)
    mappers->cleanup();
}

void TextureMapperManager::createTextureMappersForLayers(int layerCount)
{
  for (int layerIndex = 0; layerIndex < layerCount; ++layerIndex)
  {
    auto mappers =
        std::make_shared<TextureMappersForLayer>(bufferSize, layerIndex);
    mappersForLayers.push_back(mappers);
  }
}

void TextureMapperManager::initialize(
    Graphics::Gl *gl, std::shared_ptr<Graphics::FrameBufferObject> fbo,
    std::shared_ptr<ConstraintBufferObject> constraintBufferObject)
{
  for (auto mappersForLayer : mappersForLayers)
    mappersForLayer->initialize(gl, fbo);

  occlusionTexture = std::make_shared<Graphics::StandardTexture2d>(
      bufferSize, bufferSize, GL_R32F);
  occlusionTexture->initialize(gl);

  constraintTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadOnlyMapper(
          constraintBufferObject->getRenderTextureId(),
          constraintBufferObject->getWidth(),
          constraintBufferObject->getHeight()));

  occlusionTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(
          occlusionTexture->getId(), occlusionTexture->getWidth(),
          occlusionTexture->getHeight()));
}

void TextureMapperManager::resize(int width, int height)
{
  for (auto mappers : mappersForLayers)
    mappers->resize(width, height);
}

void TextureMapperManager::update()
{
  for (auto mappers : mappersForLayers)
    mappers->update();
}

void TextureMapperManager::bindOccupancyTexture(int layerIndex)
{
  mappersForLayers[layerIndex]->bindOccupancyTexture();
}

void TextureMapperManager::bindDistanceTransform(int layerIndex)
{
  mappersForLayers[layerIndex]->bindDistanceTransform();
}

void TextureMapperManager::bindApollonius(int layerIndex)
{
  mappersForLayers[layerIndex]->bindApollonius();
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getOccupancyTextureMapper(int layerIndex)
{
  return mappersForLayers[layerIndex]->getOccupancyTextureMapper();
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getDistanceTransformTextureMapper(int layerIndex)
{
  return mappersForLayers[layerIndex]->getDistanceTransformTextureMapper();
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getApolloniusTextureMapper(int layerIndex)
{
  return mappersForLayers[layerIndex]->getApolloniusTextureMapper();
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getConstraintTextureMapper()
{
  return constraintTextureMapper;
}

void TextureMapperManager::cleanup()
{
  for (auto mappers : mappersForLayers)
    mappers->cleanup();

  constraintTextureMapper.reset();
}

void TextureMapperManager::saveOccupancy()
{
  for (auto mappers : mappersForLayers)
    mappers->saveOccupancy();
}

void TextureMapperManager::saveDistanceTransform()
{
  for (auto mappers : mappersForLayers)
    mappers->saveDistanceTransform();
}

void TextureMapperManager::saveApollonius()
{
  for (auto mappers : mappersForLayers)
    mappers->saveApollonius();
}

int TextureMapperManager::getBufferSize()
{
  return bufferSize;
}

