#ifndef SRC_TEXTURE_MAPPER_MANAGER_H_

#define SRC_TEXTURE_MAPPER_MANAGER_H_

#include <memory>
#include <string>
#include <vector>
#include "./graphics/frame_buffer_object.h"
#include "./graphics/gl.h"

class CudaTextureMapper;
class CudaTexture3DMapper;
class ConstraintBufferObject;
class TextureMappersForLayer;
namespace Graphics
{
class StandardTexture2d;
}

/**
 * \brief Container for all TextureMapperForLayer%s
 *
 * It also provides methods to save a textures to the filesystem as well
 * as methods to bind the textures to render them for debugging purposes.
 */
class TextureMapperManager
{
 public:
  explicit TextureMapperManager(int bufferSize);
  ~TextureMapperManager();

  void createTextureMappersForLayers(int layerCount);
  void
  initialize(Graphics::Gl *gl, std::shared_ptr<Graphics::FrameBufferObject> fbo,
             std::shared_ptr<ConstraintBufferObject> constraintBufferObject);

  void resize(int widht, int height);

  void update();

  void bindOcclusionTexture();
  void bindIntegralCostsTexture();
  void bindSaliencyTexture();
  void bindDistanceTransform(int layerIndex);
  void bindApollonius(int layerIndex);

  std::shared_ptr<CudaTexture3DMapper> getColorTextureMapper();
  std::shared_ptr<CudaTextureMapper> getOcclusionTextureMapper();
  std::shared_ptr<CudaTextureMapper> getSaliencyTextureMapper();
  std::shared_ptr<CudaTextureMapper>
  getDistanceTransformTextureMapper(int layerIndex);
  std::shared_ptr<CudaTextureMapper> getApolloniusTextureMapper(int layerIndex);
  std::shared_ptr<CudaTextureMapper> getConstraintTextureMapper();
  std::shared_ptr<CudaTextureMapper> getAccumulatedLayersTextureMapper();
  std::shared_ptr<CudaTextureMapper> getIntegralCostsTextureMapper();

  void cleanup();

  void saveOcclusion(std::string filename);
  void saveIntegralCostsImage(std::string filename);
  void saveDistanceTransform();
  void saveApollonius();
  void saveSaliency();

  int getBufferSize();

 private:
  int bufferSize;
  int width;
  int height;
  bool saveSaliencyInNextFrame = false;
  std::vector<std::shared_ptr<TextureMappersForLayer>> mappersForLayers;

  std::shared_ptr<CudaTextureMapper> accumulatedLayersTextureMapper;
  std::shared_ptr<CudaTextureMapper> constraintTextureMapper;

  std::shared_ptr<Graphics::StandardTexture2d> integralCostsImage;
  std::shared_ptr<CudaTextureMapper> integralCostsTextureMapper;

  std::shared_ptr<Graphics::StandardTexture2d> occlusionTexture;
  std::shared_ptr<CudaTextureMapper> occlusionTextureMapper;

  std::shared_ptr<Graphics::StandardTexture2d> saliencyTexture;
  std::shared_ptr<CudaTextureMapper> saliencyTextureMapper;

  std::shared_ptr<CudaTexture3DMapper> colorTextureMapper;
};

#endif  // SRC_TEXTURE_MAPPER_MANAGER_H_
