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
 * \brief Used to create and retrieve ShaderProgram%s
 *
 * The are managed by one instance of this class so that the same
 * ShaderProgram is not instantiated multiple times.
 *
 * ShaderProgram%s are identified by an identifier which is returend
 * by the #addShader functions.
 */
class ShaderManager
{
 public:
  ShaderManager() = default;
  ~ShaderManager();

  void initialize(Gl *gl, std::shared_ptr<HABuffer> haBuffer);

  int addShader(std::string vertexShaderPath, std::string fragmentShaderPath);
  int addShader(std::string vertexShaderPath, std::string geometryShaderPath,
                std::string fragmentShaderPath);
  void bind(int id, const RenderData &renderData);
  void bindForHABuffer(int id, const RenderData &renderData);
  std::shared_ptr<ShaderProgram> getShader(int id);

 private:
  std::vector<std::shared_ptr<ShaderProgram>> shaderPrograms;
  Gl *gl;
  std::shared_ptr<HABuffer> haBuffer;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_SHADER_MANAGER_H_
