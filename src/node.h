#ifndef SRC_NODE_H_

#define SRC_NODE_H_

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <memory>
#include "./graphics/render_data.h"
#include "./math/obb.h"
#include "./graphics/gl.h"
#include "./graphics/object_manager.h"

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

  void render(Graphics::Gl *gl,
              std::shared_ptr<Graphics::ObjectManager> objectManager,
              RenderData renderData)
  {
    this->objectManager = objectManager;
    render(gl, renderData);
  }

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

  virtual void render(Graphics::Gl *gl, RenderData renderData) = 0;

  bool persistable = true;
  std::shared_ptr<Graphics::ObjectManager> objectManager;

 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
  }
};

#endif  // SRC_NODE_H_
