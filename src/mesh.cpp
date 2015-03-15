#include "./mesh.h"
#include "./gl_assert.h"
#include <QOpenGLFunctions_4_3_Core>

Mesh::Mesh(QOpenGLFunctions_4_3_Core *gl, aiMesh *mesh)
  : gl(gl)
{
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

  QOpenGLBuffer positionBuffer(QOpenGLBuffer::VertexBuffer);
  positionBuffer.create();
  positionBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
  positionBuffer.bind();
  positionBuffer.allocate(positionData, numVerts * 3 * sizeof(float));

  QOpenGLBuffer colorBuffer(QOpenGLBuffer::VertexBuffer);
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

  buffers.push_back(positionBuffer);
  buffers.push_back(colorBuffer);
}

Mesh::~Mesh()
{
}

void Mesh::prepareShaderProgram()
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

void Mesh::render(Eigen::Matrix4f projection, Eigen::Matrix4f view)
{
  shaderProgram.bind();
  auto location = shaderProgram.uniformLocation("viewProjectionMatrix");
  Eigen::Matrix4f modelViewProjection = projection * view;

  gl->glUniformMatrix4fv(location, 1, GL_FALSE, modelViewProjection.data());

  vertexArrayObject.bind();

  glAssert(glDrawArrays(GL_TRIANGLES, 0, numVerts));
}
