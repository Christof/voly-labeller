#include "./render_object.h"
#include <string>

RenderObject::RenderObject(Gl *gl, std::string vertexShaderPath,
                           std::string fragmentShaderPath)
{
  shaderProgram =
      std::make_shared<ShaderProgram>(gl, vertexShaderPath, fragmentShaderPath);

  vertexArrayObject.create();
  bind();
}

RenderObject::~RenderObject()
{
}

void RenderObject::bind()
{
  vertexArrayObject.bind();
  shaderProgram->bind();
}

void RenderObject::release()
{
  shaderProgram->release();
  vertexArrayObject.release();
}

void RenderObject::releaseBuffers()
{
  for (auto &buffer : buffers)
    buffer.release();
}

