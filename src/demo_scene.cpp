#include "./demo_scene.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <QObject>
#include <QDebug>
#include <Eigen/Core>
#include <string>
#include "./gl.h"
#include "./input/invoke_manager.h"

DemoScene::DemoScene(std::shared_ptr<InvokeManager> invokeManager)
{
  cameraController = std::shared_ptr<CameraController>(
      new CameraController(camera));

  invokeManager->addHandler("cam", cameraController.get());
}

DemoScene::~DemoScene()
{
}

void DemoScene::initialize()
{
  glAssert(gl->glClearColor(0.9f, 0.9f, 0.8f, 1.0f));

  Assimp::Importer importer;
  const std::string filename = "../assets/assets.dae";
  const aiScene *scene = importer.ReadFile(
      filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                    aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene)
  {
    qCritical() << "Could not load " << filename.c_str();
    exit(1);
  }

  for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
  {
    auto importedMesh = scene->mMeshes[meshIndex];
    auto mesh = std::unique_ptr<Mesh>(new Mesh(
        gl, importedMesh, scene->mMaterials[importedMesh->mMaterialIndex]));
    meshes.push_back(std::move(mesh));
  }
}

void DemoScene::update(double frameTime, QSet<Qt::Key> keysPressed)
{
  this->frameTime = frameTime;
  cameraController->setFrameTime(frameTime);
}

void DemoScene::render()
{
  glAssert(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  for (auto &mesh : meshes)
    mesh->render(camera.getProjectionMatrix(), camera.getViewMatrix());
}

void DemoScene::resize(int width, int height)
{
  glAssert(glViewport(0, 0, width, height));
}

