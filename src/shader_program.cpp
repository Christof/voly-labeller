#include "./shader_program.h"
#include <string>
#include <QFile>
#include <QUrl>
#include <QString>
#include <QRegularExpression>
#include <QTextStream>
#include <QDir>
#include "./gl.h"
#include <iostream>

ShaderProgram::ShaderProgram(Gl *gl, std::string vertexShaderPath,
                             std::string fragmentShaderPath)
  : gl(gl)
{
  if (!shaderProgram.addShaderFromSourceCode(
          QOpenGLShader::Vertex,
          readFileAndHandleIncludes(vertexShaderPath.c_str())))
  {
    qCritical() << "error";
  }
  if (!shaderProgram.addShaderFromSourceCode(
          QOpenGLShader::Fragment,
          readFileAndHandleIncludes(fragmentShaderPath.c_str())))
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

void ShaderProgram::release()
{
  shaderProgram.release();
}

void ShaderProgram::enableAndSetAttributes(std::string usage,
                                           int perVertexElements)
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

void ShaderProgram::setUniform(const char *name, Eigen::Vector2f vector)
{
  auto location = shaderProgram.uniformLocation(name);
  glAssert(gl->glUniform2fv(location, 1, vector.data()));
}

void ShaderProgram::setUniform(const char *name, float value)
{
  auto location = shaderProgram.uniformLocation(name);
  glAssert(gl->glUniform1f(location, value));
}

void ShaderProgram::setUniform(const char *name, int value)
{
  auto location = shaderProgram.uniformLocation(name);
  glAssert(gl->glUniform1i(location, value));
}

QString readFile(QString path)
{
  QFile file(path);

  if (!file.open(QFile::ReadOnly | QFile::Text))
    throw std::runtime_error("The file '" + path.toStdString() +
                             "' doesn't exist!");

  std::stringstream buffer;
  QTextStream in(&file);

  return in.readAll();
}

QString ShaderProgram::readFileAndHandleIncludes(QString path)
{
  auto directory = QFileInfo(path).absoluteDir().path() + "/";
  auto source = readFile(path);

  QRegularExpression regex("[ ]*#include[ ]+[\"<](.*)[\">].*");
  int searchPosition = 0;
  auto match = regex.match(source, searchPosition);
  if (match.hasMatch())
  {
    auto filename = match.captured(1);
    std::cout << filename.toStdString() << " path " << directory.toStdString() << std::endl;
    auto includeSource = readFile(directory + filename);
    source = source.replace(match.capturedStart(0), match.capturedLength(0),
        includeSource);
  }
  else
    std::cout << "no match" << std::endl;

  return source;
}

