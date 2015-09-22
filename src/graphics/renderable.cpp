#include "./renderable.h"
#include <string>
#include "./object_manager.h"
#include "./managers.h"

namespace Graphics
{

Renderable::Renderable()
{
}

Renderable::~Renderable()
{
}

void Renderable::initialize(Gl *gl, std::shared_ptr<Managers> managers)
{
  this->objectManager = managers->getObjectManager();

  objectData = createBuffers(objectManager, managers->getTextureManager(),
                             managers->getShaderManager());
}

void Renderable::render(Gl *gl, std::shared_ptr<Managers> managers,
                        const RenderData &renderData)
{
  if (!objectData.isInitialized())
    initialize(gl, managers);

  objectData.modelMatrix = renderData.modelMatrix;
  objectManager->renderLater(objectData);
}

ObjectData Renderable::getObjectData()
{
  return objectData;
}

}  // namespace Graphics
