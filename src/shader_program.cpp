#include "./shader_program.h"
#include <string>
#include <QFile>
#include <QUrl>
#include <QString>
#include <QRegularExpression>
#include <QTextStream>
#include <QDir>
#include "./gl.h"

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

int ShaderProgram::getId()
{
  return shaderProgram.programId();
}

void ShaderProgram::setUniform(const char *name, Eigen::Matrix4f matrix)
{
  glAssert(gl->glProgramUniformMatrix4fv(getId(), getLocation(name), 1,
                                         GL_FALSE, matrix.data()));
}

void ShaderProgram::setUniform(const char *name, Eigen::Vector4f vector)
{
  glAssert(
      gl->glProgramUniform4fv(getId(), getLocation(name), 1, vector.data()));
}

void ShaderProgram::setUniform(const char *name, Eigen::Vector3f vector)
{
  glAssert(
      gl->glProgramUniform3fv(getId(), getLocation(name), 1, vector.data()));
}

void ShaderProgram::setUniform(const char *name, Eigen::Vector2f vector)
{
  glAssert(
      gl->glProgramUniform2fv(getId(), getLocation(name), 1, vector.data()));
}

void ShaderProgram::setUniform(const char *name, float value)
{
  glAssert(gl->glProgramUniform1f(getId(), getLocation(name), value));
}

void ShaderProgram::setUniform(const char *name, int value)
{
  glAssert(gl->glProgramUniform1i(getId(), getLocation(name), value));
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

  QRegularExpression regex("^[ ]*#include[ ]+[\"<](.*)[\">].*");
  regex.setPatternOptions(QRegularExpression::MultilineOption);

  auto match = regex.match(source);
  while (match.hasMatch())
  {
    auto filename = match.captured(1);
    auto includePath = directory + filename;
    auto includeSource = readFile(includePath);
    includeSource = includeSource.replace(
        QRegularExpression("^[ ]*#version \\d*.*$",
                           QRegularExpression::MultilineOption),
        "");
    source = source.replace(match.capturedStart(0), match.capturedLength(0),
                            includeSource);

    qDebug() << path << "includes" << includePath;

    match = regex.match(source, match.capturedEnd(0));
  }

  return source;
}

int ShaderProgram::getLocation(const char *name)
{
  return shaderProgram.uniformLocation(name);
}

