#include "./shader_manager.h"
#include <vector>
#include <string>
#include "./shader_program.h"
#include "./gl.h"
#include "./ha_buffer.h"

namespace Graphics
{

void ShaderManager::initialize(Gl *gl, std::shared_ptr<HABuffer> haBuffer)
{
  this->gl = gl;
  this->haBuffer = haBuffer;
}

int ShaderManager::addShader(std::string vertexShaderPath,
                             std::string fragmentShaderPath)
{
  int index = 0;
  for (auto shaderProgram : shaderPrograms)
  {
    if (shaderProgram->vertexShaderPath == vertexShaderPath &&
        shaderProgram->fragmentShaderPath == fragmentShaderPath)
    {
      return index;
    }

    ++index;
  }

  shaderPrograms.push_back(std::make_shared<ShaderProgram>(gl, vertexShaderPath,
                                                           fragmentShaderPath));

  return index;
}

int ShaderManager::addShader(std::string vertexShaderPath,
                             std::string geometryShaderPath,
                             std::string fragmentShaderPath)
{
  int index = 0;
  for (auto shaderProgram : shaderPrograms)
  {
    if (shaderProgram->vertexShaderPath == vertexShaderPath &&
        shaderProgram->geometryShaderPath == geometryShaderPath &&
        shaderProgram->fragmentShaderPath == fragmentShaderPath)
    {
      return index;
    }

    ++index;
  }

  shaderPrograms.push_back(std::make_shared<ShaderProgram>(
      gl, vertexShaderPath, geometryShaderPath, fragmentShaderPath));

  return index;
}

void ShaderManager::bind(int id, const RenderData &renderData)
{
  auto shader = shaderPrograms[id];
  shader->bind();
  haBuffer->begin(shader);

  Eigen::Matrix4f viewProjectionMatrix =
      renderData.projectionMatrix * renderData.viewMatrix;
  shader->setUniform("viewProjectionMatrix", viewProjectionMatrix);
  shader->setUniform("viewMatrix", renderData.viewMatrix);
  shader->setUniform("projectionMatrix", renderData.projectionMatrix);
}

std::shared_ptr<ShaderProgram> ShaderManager::getShader(int id)
{
  return shaderPrograms[id];
}


}  // namespace Graphics
