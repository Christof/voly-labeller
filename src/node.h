#ifndef SRC_NODE_H_

#define SRC_NODE_H_

#include "./render_data.h"

/**
 * \brief
 *
 *
 */
class Node
{
 public:
  virtual ~Node()
  {
  }

  virtual void render(const RenderData &renderData) = 0;

 protected:
  Node()
  {
  }
};

#endif  // SRC_NODE_H_
