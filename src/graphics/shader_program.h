#ifndef SRC_GRAPHICS_SHADER_PROGRAM_H_

#define SRC_GRAPHICS_SHADER_PROGRAM_H_

#include <QOpenGLShaderProgram>
#include <QString>
#include <Eigen/Core>
#include <string>
#include "./buffer.h"

namespace Graphics
{

class Gl;

/**
 * \brief Encapsulates a shader program consisting of a vertex and a fragment
 * shader
 *
 *
 */
class ShaderProgram
{
 public:
  ShaderProgram(Gl *gl, std::string vertexShaderPath,
                std::string fragmentShaderPath);
  virtual ~ShaderProgram();

  void bind();
  void release();
  void enableAndSetAttributes(std::string usage, int perVertexElements);
  int getId();

  void setUniform(const char *name, Eigen::Matrix4f matrix);
  void setUniform(const char *name, Eigen::Vector4f vector);
  void setUniform(const char *name, Eigen::Vector3f vector);
  void setUniform(const char *name, Eigen::Vector2f vector);
  void setUniform(const char *name, float value);
  void setUniform(const char *name, int value);
  void setUniform(const char *name, unsigned int value);
  void setUniform(const char *name, const Graphics::Buffer &value);

  void setUniformAsVec2Array(const char *name, float *values, int count);
  void setUniformAsVec2Array(const char *name, unsigned int *values, int count);

  static QString readFileAndHandleIncludes(QString path);

 private:
  Gl *gl;
  QOpenGLShaderProgram shaderProgram;

  int getLocation(const char *name);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_SHADER_PROGRAM_H_
