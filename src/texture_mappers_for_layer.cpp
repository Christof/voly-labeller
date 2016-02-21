#include "./texture_mappers_for_layer.h"
#include "./utils/memory.h"
#include "./placement/cuda_texture_mapper.h"
#include "./placement/distance_transform.h"
#include "./placement/occupancy.h"
#include "./placement/apollonius.h"
#include "./utils/image_persister.h"
#include "./constraint_buffer_object.h"

TextureMappersForLayer::TextureMappersForLayer(int bufferSize)
  : bufferSize(bufferSize)
{
}

TextureMappersForLayer::~TextureMappersForLayer()
{
  cleanup();
}

void TextureMappersForLayer::initialize(
    Graphics::Gl *gl, std::shared_ptr<Graphics::FrameBufferObject> fbo,
    std::shared_ptr<ConstraintBufferObject> constraintBufferObject)
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

  initializeMappers(fbo, constraintBufferObject);
}

void TextureMappersForLayer::resize(int width, int height)
{
  this->width = width;
  this->height = height;
}

void TextureMappersForLayer::update()
{
  occupancy->runKernel();

  if (saveOccupancyInNextFrame)
  {
    saveOccupancyInNextFrame = false;
    occupancyTexture->save("occupancy.tiff");
  }

  distanceTransform->run();

  if (saveDistanceTransformInNextFrame)
  {
    saveDistanceTransformInNextFrame = false;
    distanceTransformTexture->save("distanceTransform.tiff");
  }

  if (saveApolloniusInNextFrame)
  {
    saveApolloniusInNextFrame = false;
    apolloniusTexture->save("apollonius.tiff");
  }
}

void TextureMappersForLayer::bindOccupancyTexture()
{
  occupancyTexture->bind();
}

void TextureMappersForLayer::bindDistanceTransform()
{
  distanceTransformTexture->bind();
}

void TextureMappersForLayer::bindApollonius()
{
  apolloniusTexture->bind();
}

std::shared_ptr<CudaTextureMapper>
TextureMappersForLayer::getOccupancyTextureMapper()
{
  return occupancyTextureMapper;
}

std::shared_ptr<CudaTextureMapper>
TextureMappersForLayer::getDistanceTransformTextureMapper()
{
  return distanceTransformTextureMapper;
}

std::shared_ptr<CudaTextureMapper>
TextureMappersForLayer::getApolloniusTextureMapper()
{
  return apolloniusTextureMapper;
}

std::shared_ptr<CudaTextureMapper>
TextureMappersForLayer::getConstraintTextureMapper()
{
  return constraintTextureMapper;
}

void TextureMappersForLayer::cleanup()
{
  occupancy.release();
  distanceTransform.release();

  colorTextureMapper.reset();
  positionsTextureMapper.reset();
  occupancyTextureMapper.reset();
  distanceTransformTextureMapper.reset();
  apolloniusTextureMapper.reset();
  constraintTextureMapper.reset();
}

void TextureMappersForLayer::saveOccupancy()
{
  saveOccupancyInNextFrame = true;
}

void TextureMappersForLayer::saveDistanceTransform()
{
  saveDistanceTransformInNextFrame = true;
}

void TextureMappersForLayer::saveApollonius()
{
  saveApolloniusInNextFrame = true;
}

void TextureMappersForLayer::initializeMappers(
    std::shared_ptr<Graphics::FrameBufferObject> fbo,
    std::shared_ptr<ConstraintBufferObject> constraintBufferObject)
{
  colorTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteMapper(fbo->getColorTextureId(0), width,
                                               height));

  positionsTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadOnlyMapper(fbo->getDepthTextureId(0), width,
                                              height));

  distanceTransformTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(
          distanceTransformTexture->getId(),
          distanceTransformTexture->getWidth(),
          distanceTransformTexture->getHeight()));

  occupancyTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(
          occupancyTexture->getId(), occupancyTexture->getWidth(),
          occupancyTexture->getHeight()));

  apolloniusTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadWriteDiscardMapper(
          apolloniusTexture->getId(), apolloniusTexture->getWidth(),
          apolloniusTexture->getHeight()));

  constraintTextureMapper = std::shared_ptr<CudaTextureMapper>(
      CudaTextureMapper::createReadOnlyMapper(
          constraintBufferObject->getRenderTextureId(),
          constraintBufferObject->getWidth(),
          constraintBufferObject->getHeight()));

  occupancy = std::make_unique<Placement::Occupancy>(positionsTextureMapper,
                                                     occupancyTextureMapper);

  distanceTransform = std::make_unique<Placement::DistanceTransform>(
      occupancyTextureMapper, distanceTransformTextureMapper);
}

