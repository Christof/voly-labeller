#ifndef SRC_NODE_H_

#define SRC_NODE_H_

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <memory>
#include "./graphics/render_data.h"
#include "./math/obb.h"
#include "./graphics/gl.h"
#include "./graphics/managers.h"
#include "./graphics/object_manager.h"
#include "./graphics/texture_manager.h"
#include "./graphics/shader_manager.h"

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

  virtual void render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData) = 0;

  const Math::Obb &getObb()
  {
    return obb;
  }

  bool isPersistable()
  {
    return persistable;
  }

  virtual void setIsVisible(bool isVisible)
  {
    this->isVisible = isVisible;
  }

  void toggleVisibility()
  {
    setIsVisible(!isVisible);
  }

 protected:
  Node()
  {
  }

  bool persistable = true;
  bool isVisible = true;
  Math::Obb obb;

 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
  }
};

#endif  // SRC_NODE_H_
