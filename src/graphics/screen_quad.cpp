#include "./screen_quad.h"
#include "./shader_program.h"
#include "./shader_manager.h"

namespace Graphics
{

ScreenQuad::ScreenQuad(std::string vertexShaderFilename,
                       std::string fragmentShaderFilename)
  : Quad(vertexShaderFilename, fragmentShaderFilename)

{
}

void ScreenQuad::initialize(Gl *gl,
                            std::shared_ptr<ObjectManager> objectManager,
                            std::shared_ptr<TextureManager> textureManager,
                            std::shared_ptr<ShaderManager> shaderManager)
{
  Renderable::initialize(gl, objectManager, textureManager, shaderManager);
  shaderProgram = shaderManager->getShader(objectData.getShaderProgramId());
}

void ScreenQuad::setUniforms(std::shared_ptr<ShaderProgram> shader,
                             const RenderData &renderData)
{
  Eigen::Matrix4f viewProjection =
      renderData.projectionMatrix * renderData.viewMatrix;
  shader->setUniform("viewProjectionMatrix", viewProjection);
  shader->setUniform("viewMatrix", renderData.viewMatrix);
  shader->setUniform("modelMatrix", renderData.modelMatrix);
}

void ScreenQuad::render(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                        const RenderData &renderData)
{
  if (!objectData.isInitialized())
    initialize(gl, objectManager, std::shared_ptr<TextureManager>(),
               std::shared_ptr<ShaderManager>());

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

ObjectData &ScreenQuad::getObjectDataReference()
{
  return objectData;
}

}  // namespace Graphics
