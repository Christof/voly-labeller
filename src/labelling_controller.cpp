#if _WIN32
#pragma warning(disable : 4267 4996)
#endif

#include "./labelling_controller.h"
#include "./labelling_coordinator.h"

LabellingController::LabellingController(
    std::shared_ptr<LabellingCoordinator> labellingCoordinator)
  : labellingCoordinator(labellingCoordinator)
{
}

void LabellingController::toggleInternalLabelling()
{
  labellingCoordinator->internalLabellingEnabled =
      !labellingCoordinator->internalLabellingEnabled;
}

void LabellingController::toggleAnchorVisibility()
{
  labellingCoordinator->toggleAnchorVisibility();
}

void LabellingController::toggleForces()
{
  labellingCoordinator->forcesEnabled = !labellingCoordinator->forcesEnabled;
}

void LabellingController::toggleOptimizeOnIdle()
{
  labellingCoordinator->optimizeOnIdle = !labellingCoordinator->optimizeOnIdle;
}

void LabellingController::toggleApollonius()
{
  labellingCoordinator->useApollonius = !labellingCoordinator->useApollonius;
}

void LabellingController::saveOcclusion()
{
  labellingCoordinator->saveOcclusion();
}

void LabellingController::saveConstraints()
{
  labellingCoordinator->saveConstraints();
}
