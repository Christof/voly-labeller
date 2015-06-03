#include "./labels_model.h"
#include "./nodes.h"
#include "./label_node.h"
#include "./labelling/label.h"
#include "./picking_controller.h"

LabelsModel::LabelsModel(std::shared_ptr<Labels> labels,
                         PickingController &pickingController)
  : labels(labels), pickingController(pickingController)
{
  labels->subscribe(std::bind(&LabelsModel::labelUpdated, this));
  connect(this, SIGNAL(labelUpdated()), this, SLOT(resetModel()),
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
  return labels->count();
}

int LabelsModel::columnCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return 1;
}

QVariant LabelsModel::data(const QModelIndex &index, int role) const
{
  if (index.row() < 0 || index.row() >= labels->count())
    return QVariant();

  auto labelsVector = labels->getLabels();
  auto label = labelsVector[index.row()];
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
  if (row < 0 || row >= labels->count())
    return;

  auto label = labels->getLabels()[row];
  if (label.text != text.toStdString())
  {
    label.text = text.toStdString();
    labels->add(label);
  }
}

void LabelsModel::changeSizeX(int row, float sizeX)
{
  if (row < 0 || row >= labels->count())
    return;

  auto label = labels->getLabels()[row];
  if (label.size.x() != sizeX)
  {
    label.size.x() = sizeX;
    labels->add(label);
  }
}

void LabelsModel::changeSizeY(int row, float sizeY)
{
  if (row < 0 || row >= labels->count())
    return;

  auto label = labels->getLabels()[row];
  if (label.size.y() != sizeY)
  {
    label.size.y() = sizeY;
    labels->add(label);
  }
}

void LabelsModel::pick(int row)
{
  if (row < 0 || row >= labels->count())
    return;

  pickingController.startPicking(labels->getLabels()[row]);
  emit startPicking();
}

void LabelsModel::addLabel()
{
  auto labelsVector = labels->getLabels();
  auto maxIdLabelNode = std::max_element(
      labelsVector.begin(), labelsVector.end(), [](Label a, Label b)
      {
        return a.id < b.id;
      });
  int maxId = maxIdLabelNode->id;
  labels->add(Label(maxId + 1, "Change text", Eigen::Vector3f(0.5f, 0, 0)));
}

