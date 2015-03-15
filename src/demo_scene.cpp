#include "./demo_scene.h"

#include <QObject>
#include <QOpenGLContext>
#include <QDebug>
#include <Eigen/Core>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "./gl_assert.h"

DemoScene::DemoScene()
  : shaderProgram(), positionBuffer(QOpenGLBuffer::VertexBuffer),
    colorBuffer(QOpenGLBuffer::VertexBuffer)
{
  keyPressedActions[Qt::Key_W] = [this]
  {
    this->camera.moveForward(this->frameTime * this->cameraSpeed);
  };
  keyPressedActions[Qt::Key_S] = [this]
  {
    this->camera.moveBackward(this->frameTime * this->cameraSpeed);
  };
  keyPressedActions[Qt::Key_A] = [this]
  {
    this->camera.strafeLeft(this->frameTime * this->cameraSpeed);
  };
  keyPressedActions[Qt::Key_D] = [this]
  {
    this->camera.strafeRight(this->frameTime * this->cameraSpeed);
  };
  keyPressedActions[Qt::Key_Q] = [this]
  {
    this->camera.changeAzimuth(this->frameTime);
  };
  keyPressedActions[Qt::Key_E] = [this]
  {
    this->camera.changeAzimuth(-this->frameTime);
  };
  keyPressedActions[Qt::Key_R] = [this]
  {
    this->camera.changeDeclination(-this->frameTime);
  };
  keyPressedActions[Qt::Key_F] = [this]
  {
    this->camera.changeDeclination(this->frameTime);
  };
}

DemoScene::~DemoScene()
{
}

void DemoScene::initialize()
{
  Assimp::Importer importer;
  const std::string filename = "../assets/assets.dae";
  const aiScene *scene = importer.ReadFile(
      filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                    aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene)
  {
    qCritical() << "Could not load " << filename.c_str();
    exit(1);
  }

  aiMesh *mesh = scene->mMeshes[0];
  numVerts = mesh->mNumFaces * 3;

  auto positionData = new float[mesh->mNumFaces * 3 * 3];
  float* positionInsertPoint = positionData;
  auto colorData = new float[mesh->mNumFaces * 3 * 4];
  auto colorInsertPoint = colorData;
  //normalArray = new float[mesh->mNumFaces * 3 * 3];
  //uvArray = new float[mesh->mNumFaces * 3 * 2];

  for (unsigned int i = 0; i < mesh->mNumFaces; i++)
  {
    const aiFace &face = mesh->mFaces[i];

    for (int j = 0; j < 3; j++)
    {

      aiVector3D pos = mesh->mVertices[face.mIndices[j]];
      memcpy(positionInsertPoint, &pos, sizeof(float) * 3);
      positionInsertPoint += 3;

      *(colorInsertPoint++) = 1.0f;
      *(colorInsertPoint++) = 0.0f;
      *(colorInsertPoint++) = 0.0f;
      *(colorInsertPoint++) = 1.0f;

      /*
      aiColor4D color = mesh->mColors[face.mIndices[j]][0];
      memcpy(colorInsertPoint, &color, sizeof(float) * 4);
      colorInsertPoint += 3;
      */
    }
  }

  prepareShaderProgram();
  // prepareVertexBuffers();
  // float colorData[] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };

  vertexArrayObject.create();
  vertexArrayObject.bind();

  positionBuffer.create();
  positionBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
  positionBuffer.bind();
  positionBuffer.allocate(positionData, numVerts * 3 * sizeof(float));

  colorBuffer.create();
  colorBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
  colorBuffer.bind();
  colorBuffer.allocate(colorData, numVerts * 4 * sizeof(float));

  shaderProgram.bind();

  positionBuffer.bind();
  shaderProgram.enableAttributeArray("vertexPosition");
  shaderProgram.setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 3);

  colorBuffer.bind();
  shaderProgram.enableAttributeArray("vertexColor");
  shaderProgram.setAttributeBuffer("vertexColor", GL_FLOAT, 0, 4);
  glCheckError();
}

void DemoScene::update(double frameTime, QSet<Qt::Key> keysPressed)
{
  this->frameTime = frameTime;
  for (Qt::Key key : keysPressed)
  {
    if (keyPressedActions.count(key))
      keyPressedActions[key]();
  }
}

void DemoScene::render()
{
  glAssert(glClear(GL_COLOR_BUFFER_BIT));

  shaderProgram.bind();
  auto location = shaderProgram.uniformLocation("viewProjectionMatrix");
  Eigen::Matrix4f modelViewProjection =
      camera.getProjectionMatrix() * camera.getViewMatrix();

  gl->glUniformMatrix4fv(location, 1, GL_FALSE, modelViewProjection.data());

  vertexArrayObject.bind();

  glAssert(glDrawArrays(GL_TRIANGLES, 0, numVerts));
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

