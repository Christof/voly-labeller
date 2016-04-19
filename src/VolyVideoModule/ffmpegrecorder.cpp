#include "ffmpegrecorder.h"
#include "ffmpeg_encoder.hpp"
#include "framepool.h"

#include <QDebug>

#if WIN32
#pragma warning(disable: 4267)
#endif

FFMPEGRecorder::FFMPEGRecorder(int width,int height, int channels, bool splithorizontal, const std::string &fn, const double fps) :
  good(false),
  initialized(false),
  error_message("not initialized"),
  video_width(width),
  video_height(height),
  video_splithorizontal(splithorizontal),
  m_file_name(""),
  m_channels(channels)

{
  if ((width %2 != 0) || (height%2 != 0))
  {
    qDebug() << "mpeg only supports frame even frame sizes!";
    ::exit(1);
  }
  video_height /= m_channels;

  m_fps = fps;

  m_file_name = fn;

  status=DVR_READY;
}

FFMPEGRecorder::~FFMPEGRecorder()
{

  for (std::vector<FramePool*>::iterator it=fp.begin(); it!= fp.end(); ++it)
  {
    (*it)->addFrame(NULL);
    if (*it)
    {
      delete (*it);
    }
  }
  qDebug() << "at end of FFMPEGRecorder::~FFMPEGRecorder()";
}

bool FFMPEGRecorder::startRecording()
{
  // creation of FFMPEG files
  if (status == DVR_RECORDING)
  {
    qDebug() << " already recording!";
    return false;
  }

  for (int i =0; i<m_channels; i++)
  {
    int fpos = m_file_name.rfind('.');
    std::string fnx;
    if (m_channels > 1)
    {
      if (fpos)
      {

        fnx = m_file_name.substr(0,fpos) + char(i+48) + m_file_name.substr(fpos);
      }
      else
      {
        fnx = m_file_name + char(i+48)+ ".mpg";
      }
    }
    else
    {
      fnx = m_file_name;
    }

    qDebug() <<" starting recording" << video_width << "x" << video_height;
    mpeg_file.push_back(new ffmpeg_encoder(video_width, video_height, video_splithorizontal, fnx .c_str(), m_fps));
  }

  nb_frames_elapsed=0;
  nb_frames_stored=0;
  nb_frames_lost_in_capture=0;
  nb_frames_lost_in_encoding=0;

  for (int i =0; i< m_channels; i++)
  {
    qDebug() <<" fpw:" << video_width << "x" << video_height;
    fp.push_back(new FramePool(video_width, video_height, 24, 20));
    fp.back()->setTopMargin(0);
    fp.back()->setBottomMargin(0);
  }

  status=DVR_RECORDING;

  // launch all the threads and we're done

  for (int i =0; i< m_channels; i++)
  {
    //m_encoding_thread.push_back(new boost::thread(FFMPEGRecorder::video_encoding, this, i));
    m_encoding_thread.push_back(new FFMPEGRecorderWorker(this, i));
  }
  return true;
}

bool FFMPEGRecorder::stopRecording() {
  if(!initialized) {
    char tmp[30];sprintf(tmp, "%d", __LINE__);
    error_message=std::string("trying to stop to record while not initialized (")+std::string(__FILE__)+":"+std::string(tmp)+")";
    return false;
  }

  if(status==DVR_RECORDING) {
    status=DVR_STOP_ASKED;
  }

  if(status==DVR_READY) {
    char tmp[30];sprintf(tmp, "%d", __LINE__);
    error_message=std::string("trying to stop to record while not recording (")+std::string(__FILE__)+":"+std::string(tmp)+")";
    return false;
  }
  for (int i =0; i< m_channels; i++)
  {
    //m_encoding_thread[i]->join();
    delete m_encoding_thread[i];
  }
  status=DVR_READY;

  int i=0;
  for (std::vector<FramePool*>::iterator it = fp.begin(); it!=fp.end(); ++it, ++i)
  {
    if (*it) {
      delete (*it); (*it)=NULL;
    }
    if (mpeg_file[i]) { delete mpeg_file[i]; mpeg_file[i]=NULL; }
  }
  return true;
}




void FFMPEGRecorder::queueFrame(unsigned char* buffer)
{
  for (int i = 0; i<m_channels; i++)
  {
    if(!fp[i]->full()) {
      fp[i]->addFrame(buffer+(video_width*video_height*3/m_channels )*i);
      nb_frames_added++;
    } else {
      nb_frames_lost_in_encoding++;
      qDebug() <<"encoding queue overflow : audio/video sync lost !!!";
    }
  }

}

