#ifndef SRC_LABEL_NODE_H_

#define SRC_LABEL_NODE_H_

#include "./node.h"
#include "./label.h"
#include "./gl.h"

/**
 * \brief Node for a label
 *
 *
 */
class LabelNode : public Node
{
 public:
  LabelNode(Label label, Gl *gl);
  virtual ~LabelNode();

  void render(const RenderData &renderData);

 private:
  Label label;
  Gl *gl;
};

#endif  // SRC_LABEL_NODE_H_
