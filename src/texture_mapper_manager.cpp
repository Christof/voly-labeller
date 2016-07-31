#if _WIN32
#pragma warning(disable : 4996 4267)
#endif

#include "./texture_mapper_manager.h"
#include <string>
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
  cleanup();
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

  integralCostsImage = std::make_shared<Graphics::StandardTexture2d>(
      bufferSize, bufferSize, GL_R32F);
  integralCostsImage->initialize(gl);

  integralCostsTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(
          integralCostsImage->getId(), integralCostsImage->getWidth(),
          integralCostsImage->getHeight()));

  occlusionTexture = std::make_shared<Graphics::StandardTexture2d>(
      bufferSize, bufferSize, GL_R32F);
  occlusionTexture->initialize(gl);

  saliencyTexture = std::make_shared<Graphics::StandardTexture2d>(
      bufferSize, bufferSize, GL_R32F);
  saliencyTexture->initialize(gl);

  accumulatedLayersTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadOnlyMapper(
          fbo->getAccumulatedLayersTextureId(), width, height));

  constraintTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadOnlyMapper(
          constraintBufferObject->getRenderTextureId(),
          constraintBufferObject->getWidth(),
          constraintBufferObject->getHeight()));

  occlusionTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(
          occlusionTexture->getId(), occlusionTexture->getWidth(),
          occlusionTexture->getHeight()));

  saliencyTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(
          saliencyTexture->getId(), saliencyTexture->getWidth(),
          saliencyTexture->getHeight()));
}

void TextureMapperManager::resize(int width, int height)
{
  this->width = width;
  this->height = height;
  for (auto mappers : mappersForLayers)
    mappers->resize(width, height);
}

void TextureMapperManager::update()
{
  if (saveSaliencyInNextFrame)
  {
    saliencyTexture->save("saliency.tiff");
    saveSaliencyInNextFrame = false;
  }

  for (auto mappers : mappersForLayers)
    mappers->update();
}

void TextureMapperManager::bindOcclusionTexture()
{
  occlusionTexture->bind();
}

void TextureMapperManager::bindSaliencyTexture()
{
  saliencyTexture->bind();
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
TextureMapperManager::getColorTextureMapper(int layerIndex)
{
  return mappersForLayers[layerIndex]->getColorTextureMapper();
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getOcclusionTextureMapper()
{
  return occlusionTextureMapper;
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getSaliencyTextureMapper()
{
  return saliencyTextureMapper;
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

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getAccumulatedLayersTextureMapper()
{
  return accumulatedLayersTextureMapper;
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getIntegralCostsTextureMapper()
{
  return integralCostsTextureMapper;
}

void TextureMapperManager::cleanup()
{
  for (auto mappers : mappersForLayers)
    mappers->cleanup();

  accumulatedLayersTextureMapper.reset();
  constraintTextureMapper.reset();
  integralCostsTextureMapper.reset();
  occlusionTextureMapper.reset();
  saliencyTextureMapper.reset();
}

void TextureMapperManager::saveOcclusion(std::string filename)
{
  occlusionTexture->save(filename);
}

void TextureMapperManager::saveSaliency()
{
  saveSaliencyInNextFrame = true;
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

