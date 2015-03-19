#ifndef SRC_SHADER_PROGRAM_H_

#define SRC_SHADER_PROGRAM_H_

#include <QOpenGLShaderProgram>
#include <Eigen/Core>
#include <string>

class QOpenGLFunctions_4_3_Core;

/**
 * \brief Encapsulates a shader program consisting of a vertex and a fragment
 * shader
 *
 *
 */
class ShaderProgram
{
 public:
  ShaderProgram(QOpenGLFunctions_4_3_Core *gl, std::string vertexShaderPath,
      std::string fragmentShaderPath);
  virtual ~ShaderProgram();

  void bind();
  void release();
  void enableAndSetAttributes(std::string usage, int perVertexElements);

  void setUniform(const char* name, Eigen::Matrix4f matrix);
  void setUniform(const char* name, Eigen::Vector4f vector);
  void setUniform(const char* name, Eigen::Vector3f vector);
  void setUniform(const char* name, float value);

 private:
  QOpenGLFunctions_4_3_Core *gl;
  QOpenGLShaderProgram shaderProgram;
};

#endif  // SRC_SHADER_PROGRAM_H_
