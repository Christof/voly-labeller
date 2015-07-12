#include "./renderable.h"
#include <string>
#include "./render_object.h"
#include "./ha_buffer.h"

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

void Renderable::render(Gl *gl, std::shared_ptr<HABuffer> haBuffer,
                        const RenderData &renderData)
{
  if (!renderObject.get())
    initialize(gl);

  renderObject->bind();

  haBuffer->begin(renderObject->shaderProgram, renderData);
  setUniforms(renderObject->shaderProgram, renderData);

  draw(gl);
  haBuffer->end();

  renderObject->release();
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
