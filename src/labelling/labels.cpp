#include "./labels.h"
#include <vector>

std::function<void()>
Labels::subscribe(std::function<void(Action action, const Label &)> subscriber)
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
  notify(Action::Add, label);
}

void Labels::update(Label label)
{
  bool notifyChanges = labels[label.id] != label;

  labels[label.id] = label;

  if (notifyChanges)
    notify(Action::Update, label);
}

void Labels::remove(Label label)
{
  labels.erase(labels.find(label.id));
  notify(Action::Delete, label);
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

void Labels::notify(Action action, const Label &label)
{
  for (auto &subscriber : subscribers)
    subscriber(action, label);
}

