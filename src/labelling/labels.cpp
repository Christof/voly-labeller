#include "./labels.h"
#include <vector>

std::function<void()>
Labels::subscribe(std::function<void(const Label &)> subscriber)
{
  int erasePosition = subscribers.size();
  subscribers.push_back(subscriber);

  return [this, erasePosition]()
  {
    subscribers.erase(subscribers.begin() + erasePosition);
  };
}


void Labels::add(Label label)
{
  labels[label.id] = label;
  notify(label);
}

void Labels::update(Label label)
{
  bool notifyChanges = labels[label.id] != label;

  labels[label.id] = label;

  if (notifyChanges)
    notify(label);
}

std::vector<Label> Labels::getLabels()
{
  std::vector<Label> result;
  for (auto &pair : labels)
    result.push_back(pair.second);

  return result;
}

Label Labels::getById(int id)
{
  return labels[id];
}

int Labels::count()
{
  return labels.size();
}

void Labels::updateAnchor(int id, Eigen::Vector3f anchorPosition)
{
  labels[id].anchorPosition = anchorPosition;
  notify(labels[id]);
}

void Labels::notify(const Label& label)
{
  for (auto &subscriber : subscribers)
    subscriber(label);
}

