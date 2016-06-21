#ifndef SRC_LABEL_NODE_H_

#define SRC_LABEL_NODE_H_

#include <QDebug>
#include <boost/serialization/nvp.hpp>
#include <memory>
#include <string>
#include "./node.h"
#include "./labelling/label.h"
#include "./graphics/quad.h"
#include "./graphics/connector.h"
#include "./graphics/mesh.h"
#include "./graphics/gl.h"
#include "./graphics/object_data.h"

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

  virtual void render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData);
  void renderLabelAndConnector(Graphics::Gl *gl,
                               std::shared_ptr<Graphics::Managers> managers,
                               RenderData renderData);

  template <class Archive> void save_construct_data(Archive &ar) const
  {
    ar << BOOST_SERIALIZATION_NVP(label);
  };

  Eigen::Vector3f labelPosition;
  Eigen::Vector3f labelPositionNDC;

  Label label;

  int layerIndex = 0;
  float anchorSize = 10.0f;

  void setIsVisible(bool isVisible);

 private:
  std::string textureText;
  Eigen::Vector2f labelSize;
  bool isVisible;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, unsigned int version) const
  {
    boost::serialization::void_cast_register<LabelNode, Node>(
        static_cast<LabelNode *>(NULL), static_cast<Node *>(NULL));
  };

  void initialize(Graphics::Gl *gl,
                  std::shared_ptr<Graphics::Managers> managers);
  void renderConnector(Graphics::Gl *gl,
                       std::shared_ptr<Graphics::Managers> managers,
                       RenderData renderData);
  void renderAnchor(Graphics::Gl *gl,
                    std::shared_ptr<Graphics::Managers> managers,
                    RenderData renderData);
  void renderLabel(Graphics::Gl *gl,
                   std::shared_ptr<Graphics::Managers> managers,
                   RenderData renderData);
  QImage *renderLabelTextToQImage();

  Eigen::Vector3f anchorNDC;
  std::shared_ptr<Graphics::Mesh> anchorMesh;
  std::shared_ptr<Graphics::Quad> quad;
  std::shared_ptr<Graphics::Connector> connector;
  int textureId = -1;
  Graphics::ObjectData labelQuad;
  Graphics::ObjectData labelConnector;
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
