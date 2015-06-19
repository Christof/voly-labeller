#ifndef SRC_SHADER_PROGRAM_H_

#define SRC_SHADER_PROGRAM_H_

#include <QOpenGLShaderProgram>
#include <QString>
#include <Eigen/Core>
#include <string>

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

  static QString readFileAndHandleIncludes(QString path);

 private:
  Gl *gl;
  QOpenGLShaderProgram shaderProgram;

  int getLocation(const char *name);
};

#endif  // SRC_SHADER_PROGRAM_H_
