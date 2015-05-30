#include "./labels_model.h"
#include "./nodes.h"
#include "./label_node.h"
#include "./label.h"
#include "./picking_controller.h"

LabelsModel::LabelsModel(std::shared_ptr<Nodes> nodes,
                         PickingController &pickingController)
  : nodes(nodes), pickingController(pickingController)
{
  connect(nodes.get(), SIGNAL(nodesChanged()), this, SLOT(resetModel()),
          Qt::QueuedConnection);
}

QHash<int, QByteArray> LabelsModel::roleNames() const
{
  QHash<int, QByteArray> roles;
  roles[NameRole] = "name";
  roles[SizeXRole] = "sizeX";
  roles[SizeYRole] = "sizeY";

  return roles;
}

int LabelsModel::rowCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return nodes->getLabelNodes().size();
}

int LabelsModel::columnCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return 1;
}

QVariant LabelsModel::data(const QModelIndex &index, int role) const
{
  auto labels = nodes->getLabelNodes();
  if (index.row() < 0 || index.row() >= static_cast<int>(labels.size()))
    return QVariant();

  auto &label = labels[index.row()]->getLabel();
  switch (role)
  {
  case NameRole:
    return QVariant::fromValue(QString(label.text.c_str()));
  case SizeXRole:
    return QVariant::fromValue(label.size.x());
  case SizeYRole:
    return QVariant::fromValue(label.size.y());
  }
  return QVariant();
}

Qt::ItemFlags LabelsModel::flags(const QModelIndex &index) const
{
  if (!index.isValid())
    return Qt::ItemIsEnabled;

  return QAbstractItemModel::flags(index) | Qt::ItemFlag::ItemIsEditable |
         Qt::ItemIsUserCheckable;
}

QVariant LabelsModel::headerData(int section, Qt::Orientation orientation,
                                 int role) const
{
  switch (role)
  {
  case NameRole:
    return QVariant::fromValue(QString("Text"));
  case SizeXRole:
    return QVariant::fromValue(QString("Width Scale"));
  case SizeYRole:
    return QVariant::fromValue(QString("Height Scale"));
  }

  return QVariant();
}

bool LabelsModel::getIsVisible() const
{
  return isVisible;
}

void LabelsModel::toggleLabelsInfoVisbility()
{
  isVisible = !isVisible;
  emit isVisibleChanged();
}

void LabelsModel::resetModel()
{
  beginResetModel();
  endResetModel();
}

void LabelsModel::changeText(int row, QString text)
{
  auto labels = nodes->getLabelNodes();
  if (row < 0 || row >= static_cast<int>(labels.size()))
    return;

  labels[row]->getLabel().text = text.toStdString();
}

void LabelsModel::changeSizeX(int row, float sizeX)
{
  auto labels = nodes->getLabelNodes();
  if (row < 0 || row >= static_cast<int>(labels.size()))
    return;

  labels[row]->getLabel().size.x() = sizeX;
}

void LabelsModel::changeSizeY(int row, float sizeY)
{
  auto labels = nodes->getLabelNodes();
  if (row < 0 || row >= static_cast<int>(labels.size()))
    return;

  labels[row]->getLabel().size.y() = sizeY;
}

void LabelsModel::pick(int row)
{
  auto labels = nodes->getLabelNodes();
  if (row < 0 || row >= static_cast<int>(labels.size()))
    return;

  pickingController.startPicking(&labels[row]->getLabel());
  emit startPicking();
}

void LabelsModel::addLabel()
{
  auto labels = nodes->getLabelNodes();
  nodes->addNode(std::make_shared<LabelNode>(
      Label(10, "Change text", Eigen::Vector3f(0, 0, 0))));
}

