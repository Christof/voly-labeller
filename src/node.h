#ifndef SRC_NODE_H_

#define SRC_NODE_H_

#include "./render_data.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>

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
 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const {};
};

#endif  // SRC_NODE_H_
