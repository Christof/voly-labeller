#include "./renderable.h"
#include <string>
#include "./render_object.h"

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

void Renderable::initialize(Gl *gl)
{
  renderObject =
      std::make_shared<RenderObject>(gl, vertexShaderPath, fragmentShaderPath);

  createBuffers(renderObject);

  renderObject->release();
  renderObject->releaseBuffers();
}

void Renderable::render(Gl *gl, const RenderData &renderData)
{
  if (!renderObject.get())
    initialize(gl);

  renderObject->bind();

  setUniforms(renderObject->shaderProgram, renderData);

  draw(gl);

  renderObject->release();
}

void Renderable::setShaderProgram(std::shared_ptr<ShaderProgram> shaderProgram)
{
  renderObject->shaderProgram = shaderProgram;
}

}  // namespace Graphics
