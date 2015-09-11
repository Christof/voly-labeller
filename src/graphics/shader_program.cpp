#include "./shader_program.h"
#include <QFile>
#include <QUrl>
#include <QString>
#include <QRegularExpression>
#include <QTextStream>
#include <QDir>
#include <string>
#include "./gl.h"

namespace Graphics
{

ShaderProgram::ShaderProgram(Gl *gl, std::string vertexShaderPath,
                             std::string fragmentShaderPath)
  : vertexShaderPath(vertexShaderPath), geometryShaderPath(""),
    fragmentShaderPath(fragmentShaderPath), gl(gl)
{
  addShaderFromSource(QOpenGLShader::Vertex, vertexShaderPath);
  addShaderFromSource(QOpenGLShader::Fragment, fragmentShaderPath);

  if (!shaderProgram.link())
  {
    throw std::runtime_error("error during linking of" + getName());
  }

  glCheckError();
}

ShaderProgram::ShaderProgram(Gl *gl, std::string vertexShaderPath,
                             std::string geometryShaderPath,
                             std::string fragmentShaderPath)
  : vertexShaderPath(vertexShaderPath), geometryShaderPath(geometryShaderPath),
    fragmentShaderPath(fragmentShaderPath), gl(gl)
{
  addShaderFromSource(QOpenGLShader::Vertex, vertexShaderPath);
  addShaderFromSource(QOpenGLShader::Geometry, geometryShaderPath);
  addShaderFromSource(QOpenGLShader::Fragment, fragmentShaderPath);

  if (!shaderProgram.link())
  {
    throw std::runtime_error("error during linking of" + getName());
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

void ShaderProgram::setUniform(const char *name, unsigned int value)
{
  glAssert(gl->glProgramUniform1ui(getId(), getLocation(name), value));
}

void ShaderProgram::setUniform(const char *name, const Graphics::Buffer &value)
{
  glAssert(gl->getShaderBufferLoad()->glProgramUniformui64NV(
      getId(), getLocation(name), value.getGpuPointer()));
}

void ShaderProgram::setUniformAsVec2Array(const char *name, float *values,
                                          int count)
{
  glAssert(gl->glProgramUniform2fv(getId(), getLocation(name), count, values));
}

void ShaderProgram::setUniformAsVec2Array(const char *name,
                                          unsigned int *values, int count)
{
  glAssert(gl->glProgramUniform2uiv(getId(), getLocation(name), count, values));
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
  if (locationCache.count(name))
    return locationCache[name];

  int location = shaderProgram.uniformLocation(name);
  if (location < 0)
    qWarning() << "Uniform" << name << "not found in" << getName().c_str();

  locationCache[name] = location;

  return location;
}

void ShaderProgram::addShaderFromSource(QOpenGLShader::ShaderType type,
                                        std::string path)
{
  if (!shaderProgram.addShaderFromSourceCode(
          type, readFileAndHandleIncludes(path.c_str())))
  {
    throw std::runtime_error("error during compilation of" + path);
  }
}

std::string ShaderProgram::getName()
{
  std::string concat =
      vertexShaderPath + "_" + geometryShaderPath + "_" + fragmentShaderPath;
  const std::string toReplace = ":/shader/";
  size_t index = 0;
  while (true)
  {
    index = concat.find(toReplace, index);
    if (index == std::string::npos)
      break;

    concat = concat.replace(index, toReplace.size(), "");
  }

  return concat;
}

}  // namespace Graphics
