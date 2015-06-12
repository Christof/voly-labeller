#include "./volume_node.h"
#include "./gl.h"
#include "./render_data.h"

VolumeNode::VolumeNode(std::string filename)
  : filename(filename)
{
}

VolumeNode::~VolumeNode()
{
}

void VolumeNode::render(Gl *gl, RenderData renderData)
{
}
