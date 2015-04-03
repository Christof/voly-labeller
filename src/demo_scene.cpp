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
  keyPressedActions[Qt::Key_S] = [this]
  {
    this->camera.moveBackward(this->frameTime * this->cameraSpeed);
  };
  keyPressedActions[Qt::Key_A] = [this]
  {
    this->camera.strafeLeft(this->frameTime * this->cameraSpeed);
  };
  keyPressedActions[Qt::Key_D] = [this]
  {
    this->camera.strafeRight(this->frameTime * this->cameraSpeed);
  };
  keyPressedActions[Qt::Key_Q] = [this]
  {
    this->camera.changeAzimuth(this->frameTime);
  };
  keyPressedActions[Qt::Key_E] = [this]
  {
    this->camera.changeAzimuth(-this->frameTime);
  };
  keyPressedActions[Qt::Key_R] = [this]
  {
    this->camera.changeDeclination(-this->frameTime);
  };
  keyPressedActions[Qt::Key_F] = [this]
  {
    this->camera.changeDeclination(this->frameTime);
  };

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
  for (Qt::Key key : keysPressed)
  {
    if (keyPressedActions.count(key))
      keyPressedActions[key]();
  }
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

