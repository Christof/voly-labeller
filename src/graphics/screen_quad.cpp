#include "./screen_quad.h"
#include <string>
#include "./shader_program.h"
#include "./shader_manager.h"
#include "./managers.h"

namespace Graphics
{

ScreenQuad::ScreenQuad(std::string vertexShaderFilename,
                       std::string fragmentShaderFilename)
  : Quad(vertexShaderFilename, fragmentShaderFilename)

{
}

void ScreenQuad::initialize(Gl *gl, std::shared_ptr<Managers> managers)
{
  Renderable::initialize(gl, managers);
  shaderProgram =
      managers->getShaderManager()->getShader(objectData.getShaderProgramId());
}

void ScreenQuad::setUniforms(std::shared_ptr<ShaderProgram> shader,
                             const RenderData &renderData)
{
  shader->setUniform("viewProjectionMatrix", renderData.viewProjectionMatrix);
  shader->setUniform("viewMatrix", renderData.viewMatrix);
  shader->setUniform("modelMatrix", renderData.modelMatrix);
}

void ScreenQuad::renderImmediately(Gl *gl, std::shared_ptr<Managers> managers,
                                   const RenderData &renderData)
{
  if (!objectData.isInitialized())
    initialize(gl, managers);

  setUniforms(shaderProgram, renderData);
  objectData.modelMatrix = renderData.modelMatrix;

  managers->getObjectManager()->renderImmediately(objectData);
}

void ScreenQuad::setShaderProgram(std::shared_ptr<ShaderProgram> shaderProgram)
{
  this->shaderProgram = shaderProgram;
}

std::shared_ptr<ShaderProgram> ScreenQuad::getShaderProgram()
{
  return this->shaderProgram;
}

ObjectData &ScreenQuad::getObjectDataReference()
{
  return objectData;
}

}  // namespace Graphics
