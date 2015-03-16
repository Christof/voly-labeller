#include "./mesh.h"
#include "./gl_assert.h"
#include <QOpenGLFunctions_4_3_Core>
#include <QDebug>

Mesh::Mesh(QOpenGLFunctions_4_3_Core *gl, aiMesh *mesh, aiMaterial *material)
  : gl(gl)
{
  numVerts = mesh->mNumFaces * 3;

  for (unsigned int i = 0; i < material->mNumProperties; ++i)
  {
    auto property = material->mProperties[i];
    std::cout << property->mKey.C_Str() << ": " << property->mType << "|"
              << property->mDataLength << std::endl;
  }
  float color[4] = { 0, 0, 0, 0 };
  unsigned int size = 4;
  if (material->Get(AI_MATKEY_COLOR_AMBIENT, color, &size) != 0)
  {
    qCritical() << "Could not load material";
    exit(1);
  }
  ambientColor = Eigen::Vector4f(color[0], color[1], color[2], color[3]);
  if (material->Get(AI_MATKEY_COLOR_DIFFUSE, color, &size) != 0)
  {
    qCritical() << "Could not load material";
    exit(1);
  }
  diffuseColor = Eigen::Vector4f(color[0], color[1], color[2], color[3]);

  std::cout << "diffuse: " << diffuseColor << " ambient: " << ambientColor << std::endl;

  auto positionData = new float[mesh->mNumFaces * 3 * 3];
  float *positionInsertPoint = positionData;
  auto normalData = new float[mesh->mNumFaces * 3 * 3];
  float *normalInsertPoint = normalData;
  auto colorData = new float[mesh->mNumFaces * 3 * 4];
  auto colorInsertPoint = colorData;
  // normalArray = new float[mesh->mNumFaces * 3 * 3];
  // uvArray = new float[mesh->mNumFaces * 3 * 2];

  for (unsigned int i = 0; i < mesh->mNumFaces; i++)
  {
    const aiFace &face = mesh->mFaces[i];

    for (int j = 0; j < 3; j++)
    {

      aiVector3D pos = mesh->mVertices[face.mIndices[j]];
      memcpy(positionInsertPoint, &pos, sizeof(float) * 3);
      positionInsertPoint += 3;

      aiVector3D normal = mesh->mNormals[face.mIndices[j]];
      memcpy(normalInsertPoint, &normal, sizeof(float) * 3);
      normalInsertPoint += 3;

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
  // float colorData[] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f
  // };

  vertexArrayObject.create();
  vertexArrayObject.bind();

  shaderProgram.bind();

  createBuffer(positionData, "vertexPosition", 3, numVerts);
  createBuffer(normalData, "vertexNormal", 3, numVerts);
  createBuffer(colorData, "vertexColor", 4, numVerts);
}

Mesh::~Mesh()
{
}

void Mesh::createBuffer(float *data, std::string usage, int perVertexElements,
                        int numberOfVertices)
{
  QOpenGLBuffer buffer(QOpenGLBuffer::VertexBuffer);
  buffer.create();
  buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
  buffer.bind();
  buffer.allocate(data, numberOfVertices * perVertexElements * sizeof(float));

  shaderProgram.enableAttributeArray(usage.c_str());
  shaderProgram.setAttributeBuffer(usage.c_str(), GL_FLOAT, 0,
                                   perVertexElements);
  glCheckError();

  buffers.push_back(buffer);
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

  location = shaderProgram.uniformLocation("ambientColor");
  glAssert(gl->glUniform4fv(location, 1, ambientColor.data()));

  location = shaderProgram.uniformLocation("diffuseColor");
  glAssert(gl->glUniform4fv(location, 1, diffuseColor.data()));

  vertexArrayObject.bind();

  glAssert(gl->glDrawArrays(GL_TRIANGLES, 0, numVerts));
}
