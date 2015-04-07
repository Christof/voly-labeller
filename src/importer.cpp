#include "./importer.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

Importer::Importer(Gl *gl) : gl(gl)
{
}

Importer::~Importer()
{
}

std::shared_ptr<Mesh> Importer::import(std::string filename, int meshIndex)
{
  Assimp::Importer importer;
  // const std::string filename = "../assets/assets.dae";
  const aiScene *scene = importer.ReadFile(
      filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                    aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene)
  {
    qCritical() << "Could not load " << filename.c_str();
    exit(1);
  }

  auto importedMesh = scene->mMeshes[meshIndex];
  return std::shared_ptr<Mesh>(new Mesh(
      gl, importedMesh, scene->mMaterials[importedMesh->mMaterialIndex]));
}
