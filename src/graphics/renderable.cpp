#include "./renderable.h"
#include <string>
#include "./render_object.h"
#include "./object_manager.h"

namespace Graphics
{

Renderable::Renderable(std::string vertexShaderPath,
                       std::string fragmentShaderPath)
  : vertexShaderPath(vertexShaderPath), fragmentShaderPath(fragmentShaderPath)
{
}

Renderable::~Renderable()
{
}

void Renderable::initialize(Gl *gl, std::shared_ptr<ObjectManager> objectManager)
{
  this->objectManager = objectManager;
  renderObject =
      std::make_shared<RenderObject>(gl, vertexShaderPath, fragmentShaderPath);

  createBuffers(renderObject, objectManager);
}

void Renderable::render(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                        const RenderData &renderData)
{
  if (!renderObject.get())
    initialize(gl, objectManager);

  // renderObject->bind();

  setUniforms(renderObject->shaderProgram, renderData);

  // draw(gl);

  // renderObject->release();
}

void Renderable::setShaderProgram(std::shared_ptr<ShaderProgram> shaderProgram)
{
  renderObject->shaderProgram = shaderProgram;
}

std::shared_ptr<ShaderProgram> Renderable::getShaderProgram()
{
  return renderObject->shaderProgram;
}

}  // namespace Graphics
