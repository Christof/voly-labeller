#ifndef SRC_LABEL_NODE_H_

#define SRC_LABEL_NODE_H_

#include <QDebug>
#include <boost/serialization/nvp.hpp>
#include <memory>
#include <string>
#include "./node.h"
#include "./label.h"

class Gl;
class Mesh;
class Quad;
class Texture;
class Connector;

/**
 * \brief Node for a label
 *
 *
 */
class LabelNode : public Node
{
 public:
  explicit LabelNode(Label label);
  virtual ~LabelNode();

  void render(Gl *gl, RenderData renderData);

  template <class Archive> void save_construct_data(Archive &ar) const
  {
    ar << BOOST_SERIALIZATION_NVP(label);
  };

  Eigen::Vector3f labelPosition;

  Label &getLabel();
 private:
  Label label;
  std::string textureText;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
    boost::serialization::void_cast_register<LabelNode, Node>(
        static_cast<LabelNode *>(NULL), static_cast<Node *>(NULL));
  };

  void renderConnector(Gl *gl, RenderData renderData);
  void renderAnchor(Gl *gl, RenderData renderData);
  void renderLabel(Gl *gl, RenderData renderData);
  QImage *renderLabelTextToQImage();

  std::shared_ptr<Mesh> anchorMesh;
  std::shared_ptr<Quad> quad;
  std::shared_ptr<Texture> texture;
  std::shared_ptr<Connector> connector;
};

namespace boost
{
namespace serialization
{

template <class Archive>
inline void save_construct_data(Archive &ar, const LabelNode *labelNode,
                                const unsigned int file_version)
{
  labelNode->save_construct_data(ar);
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
