#include "./texture_mapper_manager.h"
#include "./placement/cuda_texture_mapper.h"

TextureMapperManager::TextureMapperManager(int bufferSize)
  : bufferSize(bufferSize)
{
}

TextureMapperManager::~TextureMapperManager()
{
  cleanup();
}

void TextureMapperManager::initialize(Graphics::Gl *gl,
    std::shared_ptr<Graphics::FrameBufferObject> fbo)
{
  occupancyTexture =
      std::make_shared<Graphics::StandardTexture2d>(width, height, GL_R32F);
  occupancyTexture->initialize(gl);
  distanceTransformTexture = std::make_shared<Graphics::StandardTexture2d>(
      bufferSize, bufferSize, GL_RGBA32F);
  distanceTransformTexture->initialize(gl);


  /*
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
                                                      */
}

void TextureMapperManager::resize(int width, int height)
{
  this->width = width;
  this->height = height;
}

void TextureMapperManager::cleanup()
{
  colorTextureMapper.reset();
  positionsTextureMapper.reset();
  occupancyTextureMapper.reset();
  distanceTransformTextureMapper.reset();
}

