#include "./scene_controller.h"
#include "./scene.h"

SceneController::SceneController(std::shared_ptr<Scene> scene) : scene(scene)
{
}

void SceneController::toggleBufferViews()
{
  showBufferDebuggingViews = !showBufferDebuggingViews;
  scene->enableBufferDebuggingViews(showBufferDebuggingViews);
}

void SceneController::toggleConstraintOverlay()
{
  showConstraintOverlay = !showConstraintOverlay;
  scene->enableConstraingOverlay(showConstraintOverlay);
}

void SceneController::compositeLayers()
{
  scene->setRenderLayer(0);
}

void SceneController::renderFirstLayer()
{
  scene->setRenderLayer(1);
}

void SceneController::renderSecondLayer()
{
  scene->setRenderLayer(2);
}

