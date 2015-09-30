#ifndef SRC_DEFAULT_SCENE_CREATOR_H_

#define SRC_DEFAULT_SCENE_CREATOR_H_

#include <memory>
#include <vector>
#include "./labelling/labels.h"

class Nodes;
class Node;

/**
 * \brief Creates the default test scene, persists it and then loads it
 *
 * This is usefull to provide a default scene and also to test saving and
 * loading on every application start.
 */
class DefaultSceneCreator
{
 public:
  DefaultSceneCreator(std::shared_ptr<Nodes> nodes,
                   std::shared_ptr<Labels> labels);

  void create();

 private:
  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<Labels> labels;

  void addMeshNodesTo(std::vector<std::shared_ptr<Node>> &sceneNodes);
  void addLabelNodesTo(std::vector<std::shared_ptr<Node>> &sceneNodes);
  void addLabelsFromLabelNodes();
  void addMultiVolumeNodesTo(std::vector<std::shared_ptr<Node>> &sceneNodes);
};

#endif  // SRC_DEFAULT_SCENE_CREATOR_H_
