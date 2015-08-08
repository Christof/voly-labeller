#include "./renderable.h"
#include <string>
#include "./render_object.h"
#include "./object_manager.h"

namespace Graphics
{

Renderable::Renderable()
{
}

Renderable::~Renderable()
{
}

void Renderable::initialize(Gl *gl, std::shared_ptr<ObjectManager> objectManager)
{
  this->objectManager = objectManager;

  createBuffers(renderObject, objectManager);

  isInitialized = true;
}

void Renderable::render(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                        const RenderData &renderData)
{
  if (!isInitialized)
    initialize(gl, objectManager);

  setUniforms(std::shared_ptr<ShaderProgram>(), renderData);
}

}  // namespace Graphics
