#include "./camera_positions_model.h"
#include <iterator>
#include "./nodes.h"

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
  return cameraPositions.size();
}

int CameraPositionsModel::columnCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return 1;
}

QVariant CameraPositionsModel::data(const QModelIndex &index, int role) const
{
  if (index.row() < 0 ||
      index.row() >= static_cast<int>(cameraPositions.size()))
    return QVariant();

  if (role == CameraPositionRoles::NameRole)
    return QVariant(cameraPositions[index.row()].name.c_str());

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
  cameraNode->saveCameraPosition("Position " +
                                     std::to_string(cameraPositions.size()),
                                 cameraNode->getCamera()->getViewMatrix());
}

void CameraPositionsModel::changeName(int row, QString text)
{
  if (row < 0 || row >= static_cast<int>(cameraPositions.size()))
    return;

  ignoreNextLabelUpdate = true;
  auto cameraNode = nodes->getCameraNode();
  cameraNode->changeCameraPositionName(row, text.toStdString());
}

void CameraPositionsModel::deletePosition(int row)
{
  if (row < 0 || row >= static_cast<int>(cameraPositions.size()))
    return;

  auto cameraNode = nodes->getCameraNode();
  cameraNode->removeCameraPosition(row);
}

void CameraPositionsModel::toggleVisibility()
{
  isVisible = !isVisible;
  emit isVisibleChanged();
}

void CameraPositionsModel::moveTo(int row)
{
  auto cameraNode = nodes->getCameraNode();
  cameraNode->getCamera()->startAnimation(cameraPositions[row].viewMatrix,
                                          4.0f);
}

void CameraPositionsModel::setTo(int row)
{
  auto cameraNode = nodes->getCameraNode();
  cameraNode->getCamera()->startAnimation(cameraPositions[row].viewMatrix,
                                          1e-9f);
}

void CameraPositionsModel::update(std::vector<CameraPosition> cameraPositions)
{
  if (ignoreNextLabelUpdate)
  {
    ignoreNextLabelUpdate = false;
    return;
  }

  this->cameraPositions = cameraPositions;
  beginResetModel();
  endResetModel();
}

bool CameraPositionsModel::getIsVisible() const
{
  return isVisible;
}

