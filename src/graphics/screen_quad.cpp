#include "./screen_quad.h"
#include "./shader_program.h"

namespace Graphics
{

ScreenQuad::ScreenQuad()
{
}

void ScreenQuad::setUniforms(std::shared_ptr<ShaderProgram> shader,
                             const RenderData &renderData)
{
  Eigen::Matrix4f viewProjection =
      renderData.projectionMatrix * renderData.viewMatrix;
  shader->setUniform("viewProjectionMatrix", viewProjection);
  shader->setUniform("viewMatrix", renderData.viewMatrix);
  shader->setUniform("modelMatrix", renderData.modelMatrix);
  shader->setUniform("textureSampler", 0);
}

void ScreenQuad::render(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                        const RenderData &renderData)
{
  if (!objectData.isInitialized())
    initialize(gl, objectManager, std::shared_ptr<TextureManager>(),
               std::shared_ptr<ShaderManager>());

  if (!skipSettingUniforms)
    setUniforms(shaderProgram, renderData);

  objectManager->renderImmediately(objectData);
}

void ScreenQuad::setShaderProgram(std::shared_ptr<ShaderProgram> shaderProgram)
{
  this->shaderProgram = shaderProgram;
}

std::shared_ptr<ShaderProgram> ScreenQuad::getShaderProgram()
{
  return this->shaderProgram;
}

}  // namespace Graphics
