#include "./demo_scene.h"

#include "./gl_assert.h"

#include <QObject>
#include <QOpenGLContext>

DemoScene::DemoScene()
  : shaderProgram(), positionBuffer(QOpenGLBuffer::VertexBuffer),
    colorBuffer(QOpenGLBuffer::VertexBuffer)
{
}

DemoScene::~DemoScene()
{
}

void DemoScene::initialize()
{
  prepareShaderProgram();
  prepareVertexBuffers();
}

void DemoScene::update(float t)
{
  Q_UNUSED(t);
}

void DemoScene::render()
{
  glAssert(glClear(GL_COLOR_BUFFER_BIT));

  shaderProgram.bind();
  vertexArrayObject.bind();

  glAssert(glDrawArrays(GL_TRIANGLES, 0, 3));
}

void DemoScene::resize(int width, int height)
{
  glAssert(glViewport(0, 0, width, height));
}

void DemoScene::prepareShaderProgram()
{
  if (!shaderProgram.addShaderFromSourceFile(QOpenGLShader::Vertex,
                                              ":shader/phong.vert"))
  {
    qCritical() << "error";
  }
  if (!shaderProgram.addShaderFromSourceFile(QOpenGLShader::Fragment,
                                              ":shader/phong.frag"))
  {
    qCritical() << "error";
  }
  if (!shaderProgram.link())
  {
    qCritical() << "error";
  }
  glCheckError();
}

void DemoScene::prepareVertexBuffers()
{
  float positionData[] = { -0.8f, -0.8f, 0.0f, 0.8f, -0.8f,
                           0.0f,  0.0f,  0.8f, 0.0f };
  float colorData[] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };

  vertexArrayObject.create();
  vertexArrayObject.bind();

  positionBuffer.create();
  positionBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
  positionBuffer.bind();
  positionBuffer.allocate(positionData, 3 * 3 * sizeof(float));

  colorBuffer.create();
  colorBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
  colorBuffer.bind();
  colorBuffer.allocate(colorData, 3 * 3 * sizeof(float));

  shaderProgram.bind();

  positionBuffer.bind();
  shaderProgram.enableAttributeArray("vertexPosition");
  shaderProgram.setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 3);

  colorBuffer.bind();
  shaderProgram.enableAttributeArray("vertexColor");
  shaderProgram.setAttributeBuffer("vertexColor", GL_FLOAT, 0, 3);
  glCheckError();
}

