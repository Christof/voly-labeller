#include "./renderable.h"
#include <string>
#include "./object_manager.h"

namespace Graphics
{

Renderable::Renderable()
{
}

Renderable::~Renderable()
{
}

void Renderable::initialize(Gl *gl,
                            std::shared_ptr<ObjectManager> objectManager)
{
  this->objectManager = objectManager;

  objectData = createBuffers(objectManager, std::shared_ptr<TextureManager>(),
                             std::shared_ptr<ShaderManager>());
}

void Renderable::render(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                        std::shared_ptr<TextureManager> textureManager,
                        std::shared_ptr<ShaderManager> shaderManager,
                        const RenderData &renderData)
{
  if (!objectData.isInitialized())
    initialize(gl, objectManager);

  objectData.transform = renderData.modelMatrix;
  objectManager->renderLater(objectData);
}

}  // namespace Graphics
