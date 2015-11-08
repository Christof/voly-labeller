#include "./texture_mapper_manager.h"
#include "./utils/memory.h"
#include "./placement/cuda_texture_mapper.h"
#include "./placement/distance_transform.h"
#include "./placement/occupancy.h"
#include "./placement/apollonius.h"
#include "./utils/image_persister.h"

TextureMapperManager::TextureMapperManager(int bufferSize)
  : bufferSize(bufferSize)
{
}

TextureMapperManager::~TextureMapperManager()
{
  cleanup();
}

void TextureMapperManager::initialize(
    Graphics::Gl *gl, std::shared_ptr<Graphics::FrameBufferObject> fbo)
{
  occupancyTexture = std::make_shared<Graphics::StandardTexture2d>(
      bufferSize, bufferSize, GL_R32F);
  occupancyTexture->initialize(gl);

  distanceTransformTexture = std::make_shared<Graphics::StandardTexture2d>(
      bufferSize, bufferSize, GL_R32F);
  distanceTransformTexture->initialize(gl);

  apolloniusTexture = std::make_shared<Graphics::StandardTexture2d>(
      bufferSize, bufferSize, GL_RGBA32F);
  apolloniusTexture->initialize(gl);

  initializeMappers(fbo);
}

void TextureMapperManager::resize(int width, int height)
{
  this->width = width;
  this->height = height;
}

void TextureMapperManager::update()
{
  occupancy->runKernel();

  if (saveOccupancyInNextFrame)
  {
    saveOccupancyInNextFrame = false;
    occupancyTexture->save("occupancy.tga");
  }

  distanceTransform->run();

  if (saveDistanceTransformInNextFrame)
  {
    saveDistanceTransformInNextFrame = false;
    distanceTransformTexture->save("distanceTransform.tga");
  }

  if (saveApolloniusInNextFrame)
  {
    saveApolloniusInNextFrame = false;
    apolloniusTexture->save("apollonius.tga");
  }
}

void TextureMapperManager::bindOccupancyTexture()
{
  occupancyTexture->bind();
}

void TextureMapperManager::bindDistanceTransform()
{
  distanceTransformTexture->bind();
}

void TextureMapperManager::bindApollonius()
{
  apolloniusTexture->bind();
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getOccupancyTextureMapper()
{
  return occupancyTextureMapper;
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getDistanceTransformTextureMapper()
{
  return distanceTransformTextureMapper;
}

std::shared_ptr<CudaTextureMapper>
TextureMapperManager::getApolloniusTextureMapper()
{
  return apolloniusTextureMapper;
}

void TextureMapperManager::cleanup()
{
  occupancy.release();
  distanceTransform.release();

  colorTextureMapper.reset();
  positionsTextureMapper.reset();
  occupancyTextureMapper.reset();
  distanceTransformTextureMapper.reset();
  apolloniusTextureMapper.reset();
}

void TextureMapperManager::saveOccupancy()
{
  saveOccupancyInNextFrame = true;
}

void TextureMapperManager::saveDistanceTransform()
{
  saveDistanceTransformInNextFrame = true;
}

void TextureMapperManager::saveApollonius()
{
  saveApolloniusInNextFrame = true;
}

void TextureMapperManager::initializeMappers(
    std::shared_ptr<Graphics::FrameBufferObject> fbo)
{
  colorTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteMapper(fbo->getRenderTextureId(), width,
                                               height));

  positionsTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadOnlyMapper(fbo->getPositionTextureId(),
                                              width, height));

  distanceTransformTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(
          distanceTransformTexture->getId(), bufferSize, bufferSize));

  occupancyTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(occupancyTexture->getId(),
                                                      bufferSize, bufferSize));

  apolloniusTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(
          apolloniusTexture->getId(), bufferSize, bufferSize));

  occupancy = std::make_unique<Occupancy>(positionsTextureMapper,
                                          occupancyTextureMapper);

  distanceTransform = std::make_unique<DistanceTransform>(
      occupancyTextureMapper, distanceTransformTextureMapper);
}

