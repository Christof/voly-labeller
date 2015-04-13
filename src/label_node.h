#ifndef SRC_LABEL_NODE_H_

#define SRC_LABEL_NODE_H_

#include <QDebug>
#include <boost/serialization/nvp.hpp>
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
  LabelNode(Label label);
  virtual ~LabelNode();

  void render(Gl *gl, const RenderData &renderData);

  template <class Archive>
  void save_construct_data(Archive &ar, const LabelNode *labelNode,
                           const unsigned int file_version) const
  {
    qDebug() << "In save_construct_data method";

    ar << BOOST_SERIALIZATION_NVP(label);
  };

 private:
  Label label;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
    boost::serialization::void_cast_register<LabelNode, Node>(
        static_cast<LabelNode *>(NULL), static_cast<Node *>(NULL));
  };
};

namespace boost
{
namespace serialization
{

template <class Archive>
inline void save_construct_data(Archive &ar, const LabelNode *labelNode,
                                const unsigned int file_version)
{
  qDebug() << "In save_construct_data function";

  labelNode->save_construct_data(ar, labelNode, file_version);
}

template <class Archive>
inline void load_construct_data(Archive &ar, LabelNode *t,
                                const unsigned int version)
{
  Label label;
  ar >> BOOST_SERIALIZATION_NVP(label);

  ::new (t) LabelNode(label);
}
}  // namespace serialization
}  // namespace boost

#endif  // SRC_LABEL_NODE_H_
