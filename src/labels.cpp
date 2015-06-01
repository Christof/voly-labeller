#include "./labels.h"
#include "./label.h"

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

void Labels::notify(const Label& label)
{
  for (auto &subscriber : subscribers)
    subscriber(label);
}
