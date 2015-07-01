#include "./label_node.h"
#include <QPainter>
#include <QImage>
#include <QPoint>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "./importer.h"

LabelNode::LabelNode(Label label) : label(label)
{
  Importer importer;

  anchorMesh = importer.import("assets/anchor.dae", 0);
  quad = std::make_shared<Graphics::Quad>();

  connector = std::make_shared<Graphics::Connector>(Eigen::Vector3f(0, 0, 0),
                                          Eigen::Vector3f(1, 0, 0));
  connector->color = Eigen::Vector4f(0.75f, 0.75f, 0.75f, 1);
}

LabelNode::~LabelNode()
{
}

void LabelNode::render(Graphics::Gl *gl, RenderData renderData)
{
  if (!texture.get() || textureText != label.text)
  {
    texture = std::make_shared<Graphics::Texture>(renderLabelTextToQImage());
    texture->initialize(gl);
  }

  renderConnector(gl, renderData);
  renderAnchor(gl, renderData);
  renderLabel(gl, renderData);
}

void LabelNode::renderConnector(Graphics::Gl *gl, RenderData renderData)
{
  Eigen::Vector3f anchorToPosition = labelPosition - label.anchorPosition;
  auto length = anchorToPosition.norm();
  auto rotation = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(),
                                                     anchorToPosition);
  Eigen::Affine3f connectorTransform(
      Eigen::Translation3f(label.anchorPosition) * rotation *
      Eigen::Scaling(length));
  renderData.modelMatrix = connectorTransform.matrix();

  connector->render(gl, renderData);
}

void LabelNode::renderAnchor(Graphics::Gl *gl, RenderData renderData)
{
  Eigen::Affine3f modelTransform(Eigen::Translation3f(label.anchorPosition) *
                                 Eigen::Scaling(0.005f));
  renderData.modelMatrix = modelTransform.matrix();
  anchorMesh->render(gl, renderData);
}

void LabelNode::renderLabel(Graphics::Gl *gl, RenderData renderData)
{
  Eigen::Affine3f labelTransform(
      Eigen::Translation3f(labelPosition) *
      Eigen::Scaling(label.size.x(), label.size.y(), 1.0f));

  renderData.modelMatrix = labelTransform.matrix();

  texture->bind(gl, GL_TEXTURE0);
  quad->render(gl, renderData);
}

QImage *LabelNode::renderLabelTextToQImage()
{
  QImage *image = new QImage(512, 128, QImage::Format_ARGB32);
  image->fill(Qt::GlobalColor::transparent);

  QPainter painter;
  painter.begin(image);

  painter.setBrush(QBrush(Qt::GlobalColor::lightGray));
  painter.setPen(Qt::GlobalColor::lightGray);
  painter.drawRoundRect(QRectF(0, 0, 512, 128), 15, 60);

  painter.setPen(Qt::black);
  painter.setFont(QFont("Arial", 72));
  painter.drawText(QRectF(0, 0, 512, 128), Qt::AlignCenter, label.text.c_str());
  painter.end();

  textureText = label.text;

  return image;
}

