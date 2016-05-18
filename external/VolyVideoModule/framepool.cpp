#include <stdio.h>
#include <string.h>
#include <sys/types.h>
//#include <sys/ipc.h>
//#include <sys/sem.h>
#include "framepool.h"

FramePool::FramePool(int _width, int _height, int _bpp, int _max_frames):
  width(_width), height(_height), bpp(_bpp), opp((_bpp+7)/8), top_margin(0), bottom_margin(0),
  max_frames(_max_frames), previous_frame(NULL), newest(0), oldest(0), nb_frames(0)

{
  frame_size=width*height*opp;
  frames=new unsigned char[max_frames*frame_size];
  frames_null=new bool[max_frames];
}

FramePool::~FramePool() {
  delete []frames_null;
  delete []frames;
}

void FramePool::addFrame(unsigned char *src)
{
  QMutexLocker locker(&m_mutex);

  //printf("addFrame %d\n", nb_frames);
  unsigned char *d=frames+frame_size*newest;
  if(src) {
    memcpy(d, src, width*height*opp);

    frames_null[newest]=false;
  } else {
    frames_null[newest]=true;
  }

  previous_frame=d;

  nb_frames++;


  newest++;
  if(newest==max_frames) newest=0;

  // signal new frame available
  if (nb_frames == 1)
  {
    m_frame_available.wakeOne();
  }
}

void FramePool::addSameFrame() {
  QMutexLocker locker(&m_mutex);

  if(previous_frame) {
    unsigned char *c=frames+frame_size*newest;
    memcpy(c, previous_frame, width*height*opp);
    previous_frame=c;


    nb_frames++;

    // signal new frame available

    newest++;
    if(newest==max_frames) newest=0;
  }

  m_frame_available.wakeOne();
}

unsigned char *FramePool::getFrame()
{
  // wait for a frame to be available
  QMutexLocker locker(&m_mutex);


  if(nb_frames == 0)
  {
    m_frame_available.wait(&m_mutex);
  }

  unsigned char *c;

  if(frames_null[oldest]) {
    c=NULL;
  } else {
    c=frames+frame_size*oldest;
  }

  oldest++;
  if(oldest==max_frames) oldest=0;

  return c;
}

void FramePool::removeFrame()
{
  QMutexLocker locker(&m_mutex);
  //printf("removeFrame %d\n", nb_frames);
  nb_frames--;
}

int FramePool::size() const {
  return nb_frames;
}

int FramePool::max_size() const {
  return max_frames;
}

bool FramePool::full() const {
  return nb_frames==max_frames;
}

void FramePool::setTopMargin(int m)
{
  if(m<0) {
    top_margin=0;
  }
  else if(m>(height-bottom_margin)) {
    top_margin=height-bottom_margin;
  }
  else {
    top_margin=m;
  }
}

int FramePool::getTopMargin() const
{
  return top_margin;
}

void FramePool::setBottomMargin(int m)
{
  if(m<0) {
    bottom_margin=0;
  }
  else if(m>(height-top_margin)) {
    bottom_margin=height-top_margin;
  }
  else {
    bottom_margin=m;
  }
}

int FramePool::getBottomMargin() const
{
  return bottom_margin;
}
