#include "./mesh.h"
#include <Eigen/Geometry>
#include <boost/filesystem.hpp>
#include <QDebug>
#include <QLoggingCategory>
#include <string>
#include <vector>
#include "./gl.h"
#include "./shader_program.h"
#include "./texture_manager.h"
#include "./shader_manager.h"
#include "../utils/path_helper.h"
#include "../eigen_qdebug.h"

namespace Graphics
{

QLoggingCategory meshChan("Graphics.Mesh");

Mesh::Mesh(std::string filename, aiMesh *mesh, aiMaterial *material)
  : filename(filename)
{
  qCInfo(meshChan) << "Loading " << mesh->mName.C_Str();

  for (unsigned int i = 0; i < material->mNumProperties; ++i)
  {
    auto property = material->mProperties[i];
    qCDebug(meshChan) << property->mKey.C_Str() << ": " << property->mType
                      << "|" << property->mDataLength;
  }

  phongMaterial.ambientColor =
      loadVector4FromMaterial("$clr.ambient", material);
  phongMaterial.diffuseColor =
      loadVector4FromMaterial("$clr.diffuse", material);
  phongMaterial.specularColor =
      loadVector4FromMaterial("$clr.specular", material);
  phongMaterial.shininess = loadFloatFromMaterial("$mat.shininess", material);

  qCDebug(meshChan) << "diffuse: " << phongMaterial.diffuseColor
                    << " ambient: " << phongMaterial.ambientColor
                    << " specular: " << phongMaterial.specularColor
                    << " shininess: " << phongMaterial.shininess;

  unsigned int indicesPerFace = mesh->mFaces[0].mNumIndices;
  indexCount = indicesPerFace * mesh->mNumFaces;
  assert(indexCount > 0);

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
  if (mesh->mNormals)
  {
    memcpy(normalData, mesh->mNormals, sizeof(float) * 3 * mesh->mNumVertices);
  }
  else
  {
    qCWarning(meshChan) << "No normals in" << mesh->mName.C_Str();
  }
  textureCoordinateData = new float[mesh->mNumVertices * 2];

  hasTexture = mesh->GetNumUVChannels() > 0;
  if (hasTexture)
  {
    aiString texturePath;
    if (material->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath, nullptr,
                             nullptr, nullptr, nullptr, nullptr) == AI_SUCCESS)
    {
      std::string textureName =
          replaceBackslashesWithSlashes(texturePath.C_Str());
      auto baseFolder = boost::filesystem::path{ filename }.parent_path();
      auto texturePath = baseFolder / textureName;
      textureFilePath = texturePath.string();
      qCDebug(meshChan) << "texture" << textureFilePath.c_str();
    }
    else
    {
      qCWarning(meshChan) << "Could not load texture from material";
      hasTexture = false;
    }
  }

  if (hasTexture)
  {
    for (int i = 0; i < vertexCount; ++i)
    {
      textureCoordinateData[i * 2] = mesh->mTextureCoords[0][i].x;
      textureCoordinateData[i * 2 + 1] = -mesh->mTextureCoords[0][i].y;
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
  qCInfo(meshChan) << "Destructor of mesh" << filename.c_str();
  delete[] indexData;
  delete[] positionData;
  delete[] normalData;
  delete[] textureCoordinateData;
}

void Mesh::render(Gl *gl, std::shared_ptr<Managers> managers,
                  const RenderData &renderData)
{
  if (!hasTexture)
  {
    Eigen::Matrix4f modelViewMatrix =
        renderData.viewMatrix * objectData.modelMatrix;
    normalMatrix = modelViewMatrix.inverse().transpose();
  }

  Renderable::render(gl, managers, renderData);
}

void Mesh::createObb()
{
  Eigen::MatrixXf data(3, vertexCount);
  for (int i = 0; i < vertexCount; ++i)
    data.col(i) = Eigen::Vector3f(positionData[i * 3], positionData[i * 3 + 1],
                                  positionData[i * 3 + 2]);

  obb = Math::Obb(data);
}

ObjectData Mesh::createBuffers(std::shared_ptr<ObjectManager> objectManager,
                               std::shared_ptr<TextureManager> textureManager,
                               std::shared_ptr<ShaderManager> shaderManager)
{
  std::vector<float> pos(positionData, positionData + vertexCount * 3);
  std::vector<float> nor(normalData, normalData + vertexCount * 3);
  std::vector<float> tex(textureCoordinateData,
                         textureCoordinateData + vertexCount * 2);
  std::vector<float> col(vertexCount * 4, 1.0f);
  std::vector<unsigned int> idx(indexData, indexData + indexCount);

  int shaderProgramId = hasTexture
                            ? shaderManager->addShader(":/shader/pass.vert",
                                                       ":/shader/texture.frag")
                            : shaderManager->addShader(":/shader/phong.vert",
                                                       ":/shader/phong.frag");

  ObjectData objectData =
      objectManager->addObject(pos, nor, col, tex, idx, shaderProgramId);

  if (hasTexture)
  {
    int textureId = textureManager->addTexture(
        absolutePathOfProjectRelativePath(std::string(textureFilePath)));
    objectData.setCustomBufferFor<Graphics::TextureAddress>(
        1, [textureManager, textureId]() {
          return textureManager->getAddressFor(textureId);
        });
  }
  else
  {
    objectData.setCustomBufferFor(1, &this->phongMaterial);
    objectData.setCustomBufferFor(2, &this->normalMatrix);
  }

  return objectData;
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

}  // namespace Graphics

