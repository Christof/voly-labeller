#include "./render_object.h"
#include <string>
#include "./gl.h"

namespace Graphics
{

RenderObject::RenderObject(Gl *gl, std::string vertexShaderPath,
                           std::string fragmentShaderPath)
{
  shaderProgram =
      std::make_shared<ShaderProgram>(gl, vertexShaderPath, fragmentShaderPath);

}

RenderObject::~RenderObject()
{
}

}  // namespace Graphics

