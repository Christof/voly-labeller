#include "./camera_positions_model.h"
#include <iterator>
#include "./nodes.h"
#include "./camera_node.h"

CameraPositionsModel::CameraPositionsModel(std::shared_ptr<Nodes> nodes)
  : nodes(nodes)
{
}

QHash<int, QByteArray> CameraPositionsModel::roleNames() const
{
  QHash<int, QByteArray> roles;
  roles[NameRole] = "name";

  return roles;
}

int CameraPositionsModel::rowCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return nodes->getCameraNode()->cameraPositions.size();
}

int CameraPositionsModel::columnCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return 1;
}

QVariant CameraPositionsModel::data(const QModelIndex &index, int role) const
{
  auto positions = nodes->getCameraNode()->cameraPositions;
  if (index.row() < 0 || index.row() >= static_cast<int>(positions.size()))
    return QVariant();

  if (role == CameraPositionRoles::NameRole)
    return QVariant(positions[index.row()].name.c_str());

  return QVariant();
}

Qt::ItemFlags CameraPositionsModel::flags(const QModelIndex &index) const
{
  if (!index.isValid())
    return Qt::ItemIsEnabled;

  return QAbstractItemModel::flags(index) | Qt::ItemFlag::ItemIsEditable |
         Qt::ItemIsUserCheckable;
}

QVariant CameraPositionsModel::headerData(int section,
                                          Qt::Orientation orientation,
                                          int role) const
{
  switch (role)
  {
  case NameRole:
    return QVariant::fromValue(QString("Text"));
  }

  return QVariant();
}

void CameraPositionsModel::save()
{
  auto cameraNode = nodes->getCameraNode();
  cameraNode->cameraPositions.push_back(
      CameraPosition("New", cameraNode->getCamera()->getViewMatrix()));

  beginResetModel();
  endResetModel();
}

void CameraPositionsModel::changeName(int row, QString text)
{
  auto cameraNode = nodes->getCameraNode();
  auto &positions = cameraNode->cameraPositions;
  if (row < 0 || row >= static_cast<int>(positions.size()))
    return;

  positions[row].name = text.toStdString();
}
