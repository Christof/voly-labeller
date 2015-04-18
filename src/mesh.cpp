#include "./mesh.h"
#include <QDebug>
#include <string>
#include "./gl.h"
#include "./render_object.h"
#include "./shader_program.h"

Mesh::Mesh(aiMesh *mesh, aiMaterial *material)
{
  /*
  for (unsigned int i = 0; i < material->mNumProperties; ++i)
  {
    auto property = material->mProperties[i];
    std::cout << property->mKey.C_Str() << ": " << property->mType << "|"
              << property->mDataLength << std::endl;
  }
  */

  ambientColor = loadVector4FromMaterial("$clr.ambient", material);
  diffuseColor = loadVector4FromMaterial("$clr.diffuse", material);
  specularColor = loadVector4FromMaterial("$clr.specular", material);
  shininess = loadFloatFromMaterial("$mat.shininess", material);

  /*
  std::cout << "diffuse: " << diffuseColor << " ambient: " << ambientColor
            << " specular: " << specularColor << " shininess: " << shininess
            << std::endl;
            */

  unsigned int indicesPerFace = mesh->mFaces[0].mNumIndices;
  indexCount = indicesPerFace * mesh->mNumFaces;

  indexData = new unsigned int[indexCount];
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
  positionData = new float[mesh->mNumVertices * 3];
  memcpy(positionData, mesh->mVertices, sizeof(float) * 3 * mesh->mNumVertices);
  normalData = new float[mesh->mNumVertices * 3];
  memcpy(normalData, mesh->mNormals, sizeof(float) * 3 * mesh->mNumVertices);
}

Mesh::~Mesh()
{
}

void Mesh::initialize(Gl *gl)
{
  renderObject = std::unique_ptr<RenderObject>(
      new RenderObject(gl, ":/shader/phong.vert", ":/shader/phong.frag"));

  renderObject->createBuffer(QOpenGLBuffer::Type::IndexBuffer, indexData,
                             "index", 1, indexCount);
  delete[] indexData;

  renderObject->createBuffer(QOpenGLBuffer::Type::VertexBuffer, positionData,
                             "vertexPosition", 3, vertexCount);
  delete[] positionData;
  renderObject->createBuffer(QOpenGLBuffer::Type::VertexBuffer, normalData,
                             "vertexNormal", 3, vertexCount);
  delete[] normalData;

  renderObject->release();
  renderObject->releaseBuffers();
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

void Mesh::render(Gl *gl, const RenderData &renderData)
{
  if (!renderObject.get())
    initialize(gl);

  renderObject->bind();

  Eigen::Matrix4f modelViewProjection = renderData.projectionMatrix *
                                        renderData.viewMatrix *
                                        renderData.modelMatrix;
  renderObject->shaderProgram->setUniform("modelViewProjectionMatrix",
                                          modelViewProjection);
  renderObject->shaderProgram->setUniform("modelMatrix",
                                          renderData.modelMatrix);
  renderObject->shaderProgram->setUniform("ambientColor", ambientColor);
  renderObject->shaderProgram->setUniform("diffuseColor", diffuseColor);
  renderObject->shaderProgram->setUniform("specularColor", specularColor);
  auto view = renderData.viewMatrix;
  renderObject->shaderProgram->setUniform(
      "cameraDirection", Eigen::Vector3f(view(2, 0), view(2, 1), view(2, 2)));
  renderObject->shaderProgram->setUniform("shininess", shininess);

  glAssert(gl->glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0));

  renderObject->release();
}

