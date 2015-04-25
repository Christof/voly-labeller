#include "./label_node.h"
#include <QPainter>
#include <QImage>
#include <QPoint>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "./gl.h"
#include "./importer.h"
#include "./mesh.h"
#include "./quad.h"
#include "./texture.h"
#include "./connector.h"

LabelNode::LabelNode(Label label) : label(label)
{
  Importer importer;

  anchorMesh = importer.import("assets/anchor.dae", 0);
  quad = std::make_shared<Quad>();

  labelPosition = label.anchorPosition * 1.3f;
  connector = std::make_shared<Connector>(label.anchorPosition, labelPosition);
  connector->color = Eigen::Vector4f(0.75f, 0.75f, 0.75f, 1);
}

LabelNode::~LabelNode()
{
}

const Label& LabelNode::getLabel()
{
  return label;
}

void LabelNode::render(Gl *gl, RenderData renderData)
{
  if (!texture.get())
  {
    texture = std::make_shared<Texture>(renderLabelTextToQImage());
    texture->initialize(gl);
  }

  connector->render(gl, renderData);

  Eigen::Affine3f transform(Eigen::Translation3f(label.anchorPosition) *
                            Eigen::Scaling(0.005f));
  renderData.modelMatrix = transform.matrix();
  anchorMesh->render(gl, renderData);

  Eigen::Affine3f labelTransform(Eigen::Translation3f(labelPosition) *
                                 Eigen::Scaling(2.0f, 0.5f, 1.0f) *
                                 Eigen::Scaling(0.07f));

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

  return image;
}

