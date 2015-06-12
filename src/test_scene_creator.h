#ifndef SRC_TEST_SCENE_CREATOR_H_

#define SRC_TEST_SCENE_CREATOR_H_

#include <memory>
#include "./labelling/labels.h"

class Nodes;

/**
 * \brief Creates the default test scene, persists it and then loads it
 *
 * This is usefull to provide a default scene and also to test saving and
 * loading on every application start.
 */
class TestSceneCreator
{
 public:
  TestSceneCreator(std::shared_ptr<Nodes> nodes,
                   std::shared_ptr<Labels> labels);

  void create();

 private:
  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<Labels> labels;
};

#endif  // SRC_TEST_SCENE_CREATOR_H_
