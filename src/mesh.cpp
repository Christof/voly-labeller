#include "./mesh.h"
#include <QOpenGLFunctions_4_3_Core>
#include <QDebug>
#include <string>
#include "./gl_assert.h"

Mesh::Mesh(QOpenGLFunctions_4_3_Core *gl, aiMesh *mesh, aiMaterial *material)
  : gl(gl), shaderProgram(gl, ":shader/phong.vert", ":shader/phong.frag")
{
  for (unsigned int i = 0; i < material->mNumProperties; ++i)
  {
    auto property = material->mProperties[i];
    std::cout << property->mKey.C_Str() << ": " << property->mType << "|"
              << property->mDataLength << std::endl;
  }

  ambientColor = loadVector4FromMaterial("$clr.ambient", material);
  diffuseColor = loadVector4FromMaterial("$clr.diffuse", material);
  specularColor = loadVector4FromMaterial("$clr.specular", material);
  shininess = loadFloatFromMaterial("$mat.shininess", material);

  std::cout << "diffuse: " << diffuseColor << " ambient: " << ambientColor
            << " specular: " << specularColor << " shininess: " << shininess
            << std::endl;

  unsigned int indicesPerFace = mesh->mFaces[0].mNumIndices;
  indexCount = indicesPerFace * mesh->mNumFaces;

  unsigned int *indexData = new unsigned int[indexCount];
  auto indexInsertPoint = indexData;
  for (unsigned int i = 0; i < mesh->mNumFaces; i++)
  {
    const aiFace &face = mesh->mFaces[i];
    assert(face.mNumIndices == indicesPerFace);
    memcpy(indexInsertPoint, face.mIndices,
           sizeof(unsigned int) * face.mNumIndices);
    indexInsertPoint += face.mNumIndices;
  }

  vertexCount = mesh->mNumVertices;
  auto positionData = new float[mesh->mNumVertices * 3];
  memcpy(positionData, mesh->mVertices, sizeof(float) * 3 * mesh->mNumVertices);
  auto normalData = new float[mesh->mNumVertices * 3];
  memcpy(normalData, mesh->mNormals, sizeof(float) * 3 * mesh->mNumVertices);

  vertexArrayObject.create();
  vertexArrayObject.bind();

  shaderProgram.bind();

  createBuffer(QOpenGLBuffer::Type::IndexBuffer, indexData, "index", 1,
               indexCount);

  createBuffer(QOpenGLBuffer::Type::VertexBuffer, positionData,
               "vertexPosition", 3, vertexCount);
  createBuffer(QOpenGLBuffer::Type::VertexBuffer, normalData, "vertexNormal", 3,
               vertexCount);
}

Mesh::~Mesh()
{
}

Eigen::Vector4f Mesh::loadVector4FromMaterial(const char *key,
                                              aiMaterial *material)
{
  float values[4];
  unsigned int size = 4;
  if (material->Get(key, 0, 0, values, &size) != 0)
  {
    qCritical() << "Could not load " << key << " from material";
    exit(1);
  }

  return Eigen::Vector4f(values[0], values[1], values[2], values[3]);
}

float Mesh::loadFloatFromMaterial(const char *key, aiMaterial *material)
{
  float result;
  if (material->Get(key, 0, 0, result) != 0)
  {
    qCritical() << "Could not load " << key << " from material";
    exit(1);
  }

  return result;
}

template <class ElementType>
void Mesh::createBuffer(QOpenGLBuffer::Type bufferType, ElementType *data,
                        std::string usage, int perVertexElements,
                        int numberOfVertices)
{
  QOpenGLBuffer buffer(bufferType);
  buffer.create();
  buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
  buffer.bind();
  buffer.allocate(data,
                  numberOfVertices * perVertexElements * sizeof(ElementType));
  glCheckError();

  if (bufferType != QOpenGLBuffer::Type::IndexBuffer)
    shaderProgram.enableAndSetAttributes(usage, perVertexElements);

  buffers.push_back(buffer);
}

void Mesh::render(Eigen::Matrix4f projection, Eigen::Matrix4f view)
{
  shaderProgram.bind();

  Eigen::Matrix4f modelViewProjection = projection * view;
  shaderProgram.setUniform("viewProjectionMatrix", modelViewProjection);
  shaderProgram.setUniform("ambientColor", ambientColor);
  shaderProgram.setUniform("diffuseColor", diffuseColor);
  shaderProgram.setUniform("specularColor", specularColor);
  shaderProgram.setUniform("cameraDirection",
                           Eigen::Vector3f(view(2, 0), view(2, 1), view(2, 2)));
  shaderProgram.setUniform("shininess", shininess);

  vertexArrayObject.bind();

  glAssert(gl->glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0));
}

