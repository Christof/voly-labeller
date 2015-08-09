#ifndef SRC_GRAPHICS_SHADER_MANAGER_H_

#define SRC_GRAPHICS_SHADER_MANAGER_H_

#include <vector>
#include <memory>
#include <string>
#include "./render_data.h"

namespace Graphics
{

class ShaderProgram;
class Gl;
class HABuffer;

/**
 * \brief
 *
 *
 */
class ShaderManager
{
 public:
  ShaderManager() = default;

  void initialize(Gl *gl, std::shared_ptr<HABuffer> haBuffer);

  int addShader(std::string vertexShaderPath, std::string fragmentShaderPath);
  int addShader(std::string vertexShaderPath, std::string geometryShaderPath,
                std::string fragmentShaderPath);
  void bind(int id, const RenderData &renderData);

 private:
  std::vector<std::shared_ptr<ShaderProgram>> shaderPrograms;
  Gl *gl;
  std::shared_ptr<HABuffer> haBuffer;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_SHADER_MANAGER_H_
