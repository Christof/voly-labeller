#include "./labels_model.h"
#include "./nodes.h"
#include "./label_node.h"
#include "./label.h"

LabelsModel::LabelsModel(std::shared_ptr<Nodes> nodes)
  : nodes(nodes)
{
  connect(nodes.get(), &Nodes::nodesChanged, [this]() {
        this->beginResetModel();
        this->endResetModel();
      });
}

QHash<int, QByteArray> LabelsModel::roleNames() const
{
  QHash<int, QByteArray> roles;
  roles[NameRole] = "name";

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
  if (index.row() < 0 ||
      index.row() >= static_cast<int>(labels.size()))
    return QVariant();

  auto &label = labels[index.row()]->getLabel();
  switch (role)
  {
  case NameRole:
    return QVariant::fromValue(QString(label.text.c_str()));
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

bool LabelsModel::getIsVisible() const
{
  return isVisible;
}

void LabelsModel::toggleLabelsInfoVisbility()
{
  isVisible = !isVisible;
  emit isVisibleChanged();
}

