#include "./labeller_model.h"
#include <QColor>
#include <Eigen/Core>

LabellerModel::LabellerModel(std::shared_ptr<Forces::Labeller> labeller)
  : labeller(labeller)
{
}

QHash<int, QByteArray> LabellerModel::roleNames() const
{
  QHash<int, QByteArray> roles;
  roles[NameRole] = "name";
  roles[EnabledRole] = "enabled";
  roles[WeightRole] = "weight";
  roles[ColorRole] = "forceColor";

  return roles;
}

int LabellerModel::rowCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return labeller->forces.size();
}

int LabellerModel::columnCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return 3;
}

QVariant LabellerModel::data(const QModelIndex &index, int role) const
{
  if (index.row() < 0 ||
      index.row() >= static_cast<int>(labeller->forces.size()))
    return QVariant();

  auto &force = labeller->forces[index.row()];
  switch (role)
  {
  case NameRole:
    return QVariant::fromValue(QString(force->name.c_str()));
  case EnabledRole:
    return QVariant::fromValue(force->isEnabled);
  case WeightRole:
    return QVariant::fromValue(force->weight);
  case ColorRole:
    auto color = force->color;
    return QVariant::fromValue(
        QColor::fromRgbF(color.x(), color.y(), color.z()));
  }
  return QVariant();
}

bool LabellerModel::setData(const QModelIndex &index, const QVariant &value,
                            int role)
{
  if (role == EnabledRole || Qt::EditRole)
    labeller->forces[index.row()]->isEnabled = value.toBool();

  emit dataChanged(index, index);

  return true;
}

Qt::ItemFlags LabellerModel::flags(const QModelIndex &index) const
{
  if (!index.isValid())
    return Qt::ItemIsEnabled;

  return QAbstractItemModel::flags(index) | Qt::ItemFlag::ItemIsEditable |
         Qt::ItemIsUserCheckable;
}

QVariant LabellerModel::headerData(int section, Qt::Orientation orientation,
                                   int role) const
{
  if (role != Qt::DisplayRole)
    return QVariant();

  if (orientation == Qt::Horizontal)
    return QString("Column %1").arg(section);
  else
    return QString("Row %1").arg(section);
}

void LabellerModel::changeEnabled(int row, QVariant newValue)
{
  labeller->forces[row]->isEnabled = newValue.toBool();
}

void LabellerModel::changeWeight(int row, QVariant newValue)
{
  bool converted = false;
  auto value = newValue.toFloat(&converted);
  if (converted)
    labeller->forces[row]->weight = value;
}

void LabellerModel::toggleUpdatePositions()
{
  labeller->updatePositions = !labeller->updatePositions;
}

