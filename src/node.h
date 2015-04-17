#ifndef SRC_NODE_H_

#define SRC_NODE_H_

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include "./render_data.h"

class Gl;
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

  virtual void render(Gl *gl, RenderData renderData) = 0;

 protected:
  Node()
  {
  }

 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
  }
};

#endif  // SRC_NODE_H_
