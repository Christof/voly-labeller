#include "./texture_mapper_manager.h"
#include "./utils/memory.h"
#include "./placement/cuda_texture_mapper.h"
#include "./placement/distance_transform.h"
#include "./placement/occupancy.h"
#include "./placement/apollonius.h"

TextureMapperManager::TextureMapperManager(int bufferSize)
  : bufferSize(bufferSize)
{
}

TextureMapperManager::~TextureMapperManager()
{
  cleanup();
}

void TextureMapperManager::initialize(Graphics::Gl *gl)
{
  occupancyTexture =
      std::make_shared<Graphics::StandardTexture2d>(width, height, GL_R32F);
  occupancyTexture->initialize(gl);
  distanceTransformTexture = std::make_shared<Graphics::StandardTexture2d>(
      bufferSize, bufferSize, GL_RGBA32F);
  distanceTransformTexture->initialize(gl);
}

void TextureMapperManager::resize(int width, int height)
{
  this->width = width;
  this->height = height;
}

void
TextureMapperManager::update(std::shared_ptr<Graphics::FrameBufferObject> fbo)
{
  if (!colorTextureMapper.get())
    initializeMappers(fbo);

  occupancy->runKernel();

  distanceTransform->run();
}

void TextureMapperManager::bindOccupancyTexture()
{
  occupancyTexture->bind();
}

void TextureMapperManager::bindDistanceTransform()
{
  distanceTransformTexture->bind();
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

void TextureMapperManager::cleanup()
{
  occupancy.release();
  distanceTransform.release();

  colorTextureMapper.reset();
  positionsTextureMapper.reset();
  occupancyTextureMapper.reset();
  distanceTransformTextureMapper.reset();
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
                                                      width, height));
  occupancy = std::make_unique<Occupancy>(positionsTextureMapper,
                                          occupancyTextureMapper);
  distanceTransform = std::make_unique<DistanceTransform>(
      occupancyTextureMapper, distanceTransformTextureMapper);
}

