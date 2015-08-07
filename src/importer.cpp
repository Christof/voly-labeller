#include "./importer.h"
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <QDebug>
#include <string>
#include <vector>
#include "./utils/path_helper.h"

Importer::Importer()
{
}

Importer::~Importer()
{
}

const aiScene *Importer::readScene(std::string filename)
{
  qDebug() << "readScene:" << filename.c_str();

  if (scenes.count(filename))
    return scenes[filename];

  std::string path = absolutePathOfProjectRelativePath(filename);
  const aiScene *scene = importer.ReadFile(
      path, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene)
  {
    qCritical() << "Could not load " << path.c_str();
    exit(1);
  }

  scenes[filename] = scene;

  return scene;
}

std::shared_ptr<Graphics::Mesh> Importer::import(std::string filename,
                                                 int meshIndex)
{
  const aiScene *scene = readScene(filename);
  auto importedMesh = scene->mMeshes[meshIndex];
  return std::shared_ptr<Graphics::Mesh>(new Graphics::Mesh(
      importedMesh, scene->mMaterials[importedMesh->mMaterialIndex]));
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

