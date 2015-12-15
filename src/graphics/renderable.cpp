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
  objectData =
      createBuffers(managers->getObjectManager(), managers->getTextureManager(),
                    managers->getShaderManager());
}

void Renderable::render(Gl *gl, std::shared_ptr<Managers> managers,
                        const RenderData &renderData)
{
  if (!objectData.isInitialized())
    initialize(gl, managers);

  objectData.modelMatrix = renderData.modelMatrix;
  managers->getObjectManager()->renderLater(objectData);
}

void Renderable::renderImmediately(Gl *gl, std::shared_ptr<Managers> managers,
                        const RenderData &renderData)
{
  if (!objectData.isInitialized())
    initialize(gl, managers);

  objectData.modelMatrix = renderData.modelMatrix;
  managers->getObjectManager()->renderImmediately(objectData);
}

ObjectData Renderable::getObjectData()
{
  return objectData;
}

}  // namespace Graphics
