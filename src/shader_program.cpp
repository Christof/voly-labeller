#include "./shader_program.h"
#include <QOpenGLFunctions_4_3_Core>
#include "./gl_assert.h"

ShaderProgram::ShaderProgram(QOpenGLFunctions_4_3_Core *gl,
                             std::string vertexShaderPath,
                             std::string fragmentShaderPath)
  : gl(gl)
{
  if (!shaderProgram.addShaderFromSourceFile(QOpenGLShader::Vertex,
                                             vertexShaderPath.c_str()))
  {
    qCritical() << "error";
  }
  if (!shaderProgram.addShaderFromSourceFile(QOpenGLShader::Fragment,
                                             fragmentShaderPath.c_str()))
  {
    qCritical() << "error";
  }
  if (!shaderProgram.link())
  {
    qCritical() << "error";
  }
  glCheckError();
}

ShaderProgram::~ShaderProgram()
{
}

void ShaderProgram::bind()
{
  shaderProgram.bind();
}

void ShaderProgram::enableAndSetAttributes(std::string usage, int perVertexElements)
{
  shaderProgram.enableAttributeArray(usage.c_str());
  shaderProgram.setAttributeBuffer(usage.c_str(), GL_FLOAT, 0,
                                   perVertexElements);
  glCheckError();
}

void ShaderProgram::setUniform(const char *name, Eigen::Matrix4f matrix)
{
  auto location = shaderProgram.uniformLocation(name);
  glAssert(gl->glUniformMatrix4fv(location, 1, GL_FALSE, matrix.data()));
}

void ShaderProgram::setUniform(const char *name, Eigen::Vector4f vector)
{
  auto location = shaderProgram.uniformLocation(name);
  glAssert(gl->glUniform4fv(location, 1, vector.data()));
}

void ShaderProgram::setUniform(const char *name, Eigen::Vector3f vector)
{
  auto location = shaderProgram.uniformLocation(name);
  glAssert(gl->glUniform3fv(location, 1, vector.data()));
}

void ShaderProgram::setUniform(const char *name, float value)
{
  auto location = shaderProgram.uniformLocation(name);
  glAssert(gl->glUniform1f(location, value));
}

