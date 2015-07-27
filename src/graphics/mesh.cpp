#include "./mesh.h"
#include <QDebug>
#include <string>
#include "./gl.h"
#include "./render_object.h"
#include "./shader_program.h"
#include "./object_manager.h"
#include <iostream>

namespace Graphics
{

Mesh::Mesh(aiMesh *mesh, aiMaterial *material)
  : Renderable(":/shader/pass.vert", ":/shader/test.frag")
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
  textureCoordinateData = new float[mesh->mNumVertices * 2];

  if (mesh->GetNumUVChannels() > 0)
  {
    for (int i = 0; i < vertexCount; ++i)
    {
      textureCoordinateData[i * 2] = mesh->mTextureCoords[0][i].x;
      textureCoordinateData[i * 2 + 1] = mesh->mTextureCoords[0][i].y;
    }
  }
  else
  {
    for (int i = 0; i < vertexCount; ++i)
    {
      textureCoordinateData[i * 2] = 0.0f;
      textureCoordinateData[i * 2 + 1] = 0.0f;
    }
  }

  createObb();
}

Mesh::~Mesh()
{
  delete[] positionData;
}

void Mesh::createObb()
{
  Eigen::MatrixXf data(3, vertexCount);
  for (int i = 0; i < vertexCount; ++i)
    data.col(i) = Eigen::Vector3f(positionData[i * 3], positionData[i * 3 + 1],
                                  positionData[i * 3 + 2]);

  obb = std::make_shared<Math::Obb>(data);
}

void Mesh::createBuffers(std::shared_ptr<RenderObject> renderObject,
    std::shared_ptr<ObjectManager> objectManager)
{
  std::vector<float> pos(positionData, positionData + vertexCount * 3);
  std::vector<float> nor(normalData, normalData + vertexCount * 3);
  std::vector<float> tex(textureCoordinateData,
                         textureCoordinateData + vertexCount * 2);
  std::vector<float> col(vertexCount * 4, 0.8f);
  std::vector<unsigned int> idx(indexData, indexData + indexCount);
  id = objectManager->addObject(pos, nor, col, tex, idx);
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

void Mesh::setUniforms(std::shared_ptr<ShaderProgram> shader,
                       const RenderData &renderData)
{
  objectManager->renderLater(id, renderData.modelMatrix);
  /*
  Eigen::Matrix4f modelViewProjection = renderData.projectionMatrix *
                                        renderData.viewMatrix *
                                        renderData.modelMatrix;
  shader->setUniform("modelViewProjectionMatrix", modelViewProjection);
  shader->setUniform("modelMatrix", renderData.modelMatrix);
  shader->setUniform("ambientColor", ambientColor);
  shader->setUniform("diffuseColor", diffuseColor);
  shader->setUniform("specularColor", specularColor);
  auto view = renderData.viewMatrix;
  shader->setUniform("cameraDirection",
                     Eigen::Vector3f(view(2, 0), view(2, 1), view(2, 2)));
  shader->setUniform("lightPosition", renderData.cameraPosition);
  shader->setUniform("shininess", shininess);
  */
}

void Mesh::draw(Gl *gl)
{
  // glAssert(gl->glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0));
}

}  // namespace Graphics

