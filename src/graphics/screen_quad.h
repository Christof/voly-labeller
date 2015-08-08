#ifndef SRC_GRAPHICS_SCREEN_QUAD_H_

#define SRC_GRAPHICS_SCREEN_QUAD_H_

#include <memory>
#include "./quad.h"

namespace Graphics
{

class Gl;

/**
 * \brief A screen filling quad which is rendered to the currently bound frame
 * buffer and not to an HABuffer
 */
class ScreenQuad : public Quad
{
 public:
  ScreenQuad();
  ScreenQuad(std::string vertexShaderFilename,
             std::string fragmentShaderFilename);

  virtual void render(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                      const RenderData &renderData);

  virtual void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData);

  void setShaderProgram(std::shared_ptr<ShaderProgram> shaderProgram);
  std::shared_ptr<ShaderProgram> getShaderProgram();

  bool skipSettingUniforms = false;

 private:
  std::shared_ptr<ShaderProgram> shaderProgram;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_SCREEN_QUAD_H_
