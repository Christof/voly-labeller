#ifndef SRC_GRAPHICS_SHADER_MANAGER_H_

#define SRC_GRAPHICS_SHADER_MANAGER_H_

#include <vector>
#include <memory>
#include <string>

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

  int addShader(std::string fragmentShaderPath, std::string vertexShaderPath);
  std::shared_ptr<ShaderProgram> get(int id);

 private:
  std::vector<std::shared_ptr<ShaderProgram>> shaderPrograms;
  Gl *gl;
  std::shared_ptr<HABuffer> haBuffer;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_SHADER_MANAGER_H_
