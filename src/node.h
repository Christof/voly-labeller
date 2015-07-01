#ifndef SRC_NODE_H_

#define SRC_NODE_H_

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <memory>
#include "./render_data.h"
#include "./math/obb.h"
#include "./graphics/gl.h"


/**
 * \brief Base class for nodes which are managed by the
 * Nodes class
 *
 * The only virtual method which must be implemented is
 * Node::render.
 */
class Node
{
 public:
  virtual ~Node()
  {
  }

  virtual void render(Graphics::Gl *gl, RenderData renderData) = 0;

  virtual std::shared_ptr<Math::Obb> getObb()
  {
    return std::shared_ptr<Math::Obb>();
  }

  bool isPersistable()
  {
    return persistable;
  }

 protected:
  Node()
  {
  }

  bool persistable = true;

 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
  }
};

#endif  // SRC_NODE_H_
