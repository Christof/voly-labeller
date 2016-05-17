#ifndef FRAMEPOOL_H
#define FRAMEPOOL_H

#include <QMutex>
#include <QWaitCondition>

class FramePool
{
public:
  FramePool(int width, int height, int bpp, int max_frames=20);
  ~FramePool();

  void            addFrame(unsigned char *src);                   // add a frame in a pool. overwrite the oldest if the pool is full
  void            addSameFrame();                                 // add the same frame that has been previiously added (if it exists)
  void            removeFrame();                                  // remove the oldest frame
  unsigned char   *getFrame();                                    // return the adress of the oldest frame, wait for a frame to be added if the pool is empty

  void    setTopMargin(int);                              // horizontal margins
  int     getTopMargin() const;
  void    setBottomMargin(int);
  int     getBottomMargin() const;

  int     size() const;
  int     max_size() const;
  bool    full() const;

protected:
  enum { SEM_ID_FRAME_AVAILABLE, SEM_ID_MUTEX };
  int     width, height, bpp, opp;        // opp=(bpp+7)/8    bpp=bits per pixel,  opp=byte per pixel
  int     top_margin, bottom_margin;

  unsigned char   *frames;                                // allocation pool:
  bool *frames_null;
  int             max_frames;                             // max number of frames in the pool
  unsigned char   *previous_frame;

  int             frame_size;                             // size of a chunk in byte

  int             newest;                                 // where the next added chunk will be
  int             oldest;                                 // where the oldest chunk is

  int             nb_frames;                              // number of chunks actually in the pool

  QMutex m_mutex;
  QWaitCondition m_frame_available;
};


#endif // FRAMEPOOL_H
