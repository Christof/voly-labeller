#include "./importer.h"
#include <assimp/scene.h>
#include <assimp/postprocess.h>

Importer::Importer(Gl *gl) : gl(gl)
{
}

Importer::~Importer()
{
}

const aiScene *Importer::readScene(std::string filename)
{
  if (scenes.count(filename))
    return scenes[filename];

  const aiScene *scene = importer.ReadFile(
      filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                    aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene)
  {
    qCritical() << "Could not load " << filename.c_str();
    exit(1);
  }

  scenes[filename] = scene;

  return scene;
}

std::shared_ptr<Mesh> Importer::import(std::string filename, int meshIndex)
{
  const aiScene *scene = readScene(filename);
  auto importedMesh = scene->mMeshes[meshIndex];
  return std::shared_ptr<Mesh>(new Mesh(
      gl, importedMesh, scene->mMaterials[importedMesh->mMaterialIndex]));
}
