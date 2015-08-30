#ifndef SRC_GRAPHICS_SCREEN_QUAD_H_

#define SRC_GRAPHICS_SCREEN_QUAD_H_

#include <memory>
#include <string>
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

  virtual void initialize(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                  std::shared_ptr<TextureManager> textureManager,
                  std::shared_ptr<ShaderManager> shaderManager);
  virtual void render(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                      const RenderData &renderData);


  void setShaderProgram(std::shared_ptr<ShaderProgram> shaderProgram);
  std::shared_ptr<ShaderProgram> getShaderProgram();

  bool skipSettingUniforms = false;

 private:
  void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData);
  std::shared_ptr<ShaderProgram> shaderProgram;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_SCREEN_QUAD_H_
