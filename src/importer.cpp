#include "./importer.h"
#include <assimp/scene.h>
#include <assimp/matrix4x4.h>
#include <assimp/postprocess.h>
#include <QDebug>
#include <QLoggingCategory>
#include <string>
#include <vector>
#include "./utils/path_helper.h"
#include "./eigen_qdebug.h"

QLoggingCategory importerChan("importer");

Importer::Importer()
{
}

Importer::~Importer()
{
}

const aiScene *Importer::readScene(std::string filename)
{
  qCInfo(importerChan) << "readScene:" << filename.c_str();

  if (scenes.count(filename))
    return scenes[filename];

  std::string path = absolutePathOfProjectRelativePath(filename);
  const aiScene *scene = importer.ReadFile(
      path, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene)
  {
    qCCritical(importerChan) << "Could not load " << path.c_str();
    exit(1);
  }

  scenes[filename] = scene;

  return scene;
}

Eigen::Matrix4f toEigen(aiMatrix4x4 matrix)
{
  Eigen::Matrix4f result;
  result << matrix.a1, matrix.a2, matrix.a3, matrix.a4, matrix.b1, matrix.b2,
      matrix.b3, matrix.b4, matrix.c1, matrix.c2, matrix.c3, matrix.c4,
      matrix.d1, matrix.d2, matrix.d3, matrix.d4;

  return result;
}

bool findTransformationForMesh(aiNode *node, unsigned int meshIndex,
                               Eigen::Matrix4f &accumulator)
{
  for (unsigned int i = 0; i < node->mNumMeshes; ++i)
  {
    if (node->mMeshes[i] == static_cast<unsigned int>(meshIndex))
    {
      accumulator *= toEigen(node->mTransformation);
      return true;
    }
  }

  for (unsigned int nodeIndex = 0; nodeIndex < node->mNumChildren; ++nodeIndex)
  {
    if (findTransformationForMesh(node->mChildren[nodeIndex], meshIndex,
                                  accumulator))
    {
      accumulator = toEigen(node->mTransformation) * accumulator;
      return true;
    }
  }

  return false;
}

Eigen::Matrix4f Importer::getTransformationFor(std::string filename,
                                               int meshIndex)
{
  const aiScene *scene = readScene(filename);

  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  bool found = findTransformationForMesh(
      scene->mRootNode, static_cast<unsigned int>(meshIndex), transformation);
  qCDebug(importerChan) << "Transformation for" << filename.c_str() << meshIndex
                        << found << transformation;

  return transformation;
}

std::shared_ptr<Graphics::Mesh> Importer::import(std::string filename,
                                                 int meshIndex)
{
  const aiScene *scene = readScene(filename);
  auto importedMesh = scene->mMeshes[meshIndex];

  return std::shared_ptr<Graphics::Mesh>(new Graphics::Mesh(
      filename, importedMesh, scene->mMaterials[importedMesh->mMaterialIndex]));
}

std::vector<std::shared_ptr<Graphics::Mesh>>
Importer::importAll(std::string filename)
{
  const aiScene *scene = readScene(filename);

  std::vector<std::shared_ptr<Graphics::Mesh>> result;
  for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
    result.push_back(import(filename, i));

  return result;
}

