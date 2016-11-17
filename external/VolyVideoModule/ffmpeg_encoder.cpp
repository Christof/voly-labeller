#include "ffmpeg_encoder.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/imgutils.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>

#ifdef __cplusplus
}
#endif

#include <QDebug>


bool ffmpeg_encoder::isStaticInit = false;

ffmpeg_encoder::ffmpeg_encoder(int width, int height, const QString &filename, const double fps) :
    m_width(width),
    m_height(height),
    m_filename(filename),
    m_framecount(0),
    m_codec(NULL),
    m_context(NULL),
    m_out_size(0),
    m_file(NULL),
    m_picture(NULL)

{
  staticInit();

  /* find the mpeg1 video encoder */
  //m_codec = avcodec_find_encoder(CODEC_ID_MPEG2VIDEO);
  //m_codec = avcodec_find_encoder(AV_CODEC_ID_H);
  //m_codec = avcodec_find_encoder_by_name("libx264");
  // RGB frames instead
  m_codec = avcodec_find_encoder_by_name("libx264rgb");

  if (!m_codec) {
    fprintf(stderr, "codec libx264rgb not found\n");
    m_codec = avcodec_find_encoder(AV_CODEC_ID_MPEG2VIDEO);
    if (!m_codec)
    {
      fprintf(stderr, "codec mpeg2 not found\n");
      ::exit(1);
    }
  }

  m_context = avcodec_alloc_context3(NULL);

  /* put sample parameters */
  //m_context->bit_rate = 100000000;
  m_context->bit_rate =  20000000;

  /* resolution must be a multiple of two */
  m_context->width = (m_width/2)*2;
  m_context->height = (m_height/2)*2;
  printf("capture size: %d x %d of (%d x %d)\n", m_context->width, m_context->height, m_width, m_height);

  /* frames per second */
  m_context->time_base= (AVRational){1,25};

  m_context->gop_size = 12; /* emit one intra frame every twelve frames */
  m_context->max_b_frames=1;
  m_context->me_range = 16;
  m_context->max_qdiff = 4;
  m_context->qmin = 10;
  m_context->qmax = 51;

  // output videl pixel formats in ascending video quality order - RGB is best
  m_context->pix_fmt =  AV_PIX_FMT_YUV420P;
  //m_context->pix_fmt =  PIX_FMT_YUV422P;
  //m_context->pix_fmt = AV_PIX_FMT_YUV444P;
  //m_context->pix_fmt = AV_PIX_FMT_RGB24;

  // H264 settings
  if (m_codec->id == AV_CODEC_ID_H264)
  {
    m_context->qcompress = 0.6;
    m_context->trellis=0;
    m_context->refs = 5;
    m_context->coder_type = 0;

    m_context->me_subpel_quality = 6;
    m_context->me_cmp|= 1;

    m_context->scenechange_threshold = 40;
    m_context-> rc_buffer_size = 0;
    m_context->pix_fmt = AV_PIX_FMT_RGB24;
    m_context->level = 30;
    av_opt_set(m_context->priv_data,"subq","6",0);
    av_opt_set(m_context->priv_data,"crf","15.0",0);
    av_opt_set(m_context->priv_data,"weighted_p_pred","0",0);
    av_opt_set(m_context->priv_data,"profile","high",AV_OPT_SEARCH_CHILDREN);
    //The setting below can be used to adjust CPU usage: fast -> lower CPU usage, bigger files
    av_opt_set(m_context->priv_data,"preset","medium",0);
    av_opt_set(m_context->priv_data,"tune","zerolatency",0);
    av_opt_set(m_context->priv_data,"x264opts","rc-lookahead=0",0);
  }

  /* allocate frame/picture */
  m_picture = av_frame_alloc();
  m_picture->pts = 0;
  m_picture->format = m_context->pix_fmt;
  m_picture->width = m_context->width;
  m_picture->height = m_context->height;

  /* open codec */
  if (avcodec_open2(m_context, m_codec,NULL) < 0) {
    fprintf(stderr, "could not open codec\n");
    ::exit(1);
  }

  /*open file*/
   m_file = fopen(m_filename.toLocal8Bit(), "wb");
  if (!m_file) {
    qDebug() << "could not open " << filename;
    ::exit(1);
  }

  /* allocate raw picture buffer  buffer */

  int ret = av_image_alloc(m_picture->data, m_picture->linesize, m_context->width, m_context->height, m_context->pix_fmt, 32);
  if (ret < 0)
  {
    printf("Could not allocate raw picture buffer\n");
    ::exit(1);
  }

  /* set up input image stride */
  m_rgb_src[0] = m_rgb_src[1] = m_rgb_src[2] = NULL;  
  m_rgb_stride[0] = -width*3;
  m_rgb_stride[1] = m_rgb_stride[2] = 0;

  /* set pixel format conversion context */

  m_sws = sws_getContext(m_context->width, m_context->height, AV_PIX_FMT_RGB24,
                         m_context->width, m_context->height, m_context->pix_fmt,
                         SWS_POINT,
                         NULL, NULL, NULL);
  if (m_sws == NULL)
  {
    fprintf(stderr, "Failed to get %d ---> %d\n", AV_PIX_FMT_RGB24, m_context->pix_fmt);
    ::exit(1);
  }
}

ffmpeg_encoder::~ffmpeg_encoder()
{

  int got_output;
  /* get the delayed frames */
  for(got_output =1; got_output; m_framecount++)
  {
    fflush(stdout);

    m_picture->pts = m_framecount;

    int ret = avcodec_encode_video2(m_context, &m_packet, m_picture, &got_output);

    if (ret < 0)
    {
      fprintf(stderr, "Error encoding frame %3d\n", m_framecount);
    }
    if (got_output)
    {
      printf("write frame %3d (size=%5d)\n", m_framecount, m_packet.size);
      fwrite(m_packet.data, 1, m_packet.size, m_file);
      av_packet_unref(&m_packet);
    }
  }

  /* add sequence end code to have a real mpeg file */
  uint8_t endcode[] = { 0, 0, 1, 0xb7 };
  fwrite(endcode, 1, sizeof(endcode), m_file);
  fclose(m_file);



  avcodec_close(m_context);
  av_free(m_context);
  av_freep(&m_picture->data[0]);
  av_frame_free(&m_picture);
  sws_freeContext(m_sws);
}

void ffmpeg_encoder::addFrame(unsigned char* buffer)
{
  // set source pointer
  m_rgb_src[0] = buffer + m_context->width*m_context->height*3 - m_context->width*3;;

  // convert to target pixel format
  sws_scale(m_sws,
            m_rgb_src, m_rgb_stride, 0, m_context->height, // src stuff
            m_picture->data, m_picture->linesize);         // dst stuff


  m_picture->pts = m_framecount;

  /* encode the image */

  av_init_packet(&m_packet);
  m_packet.data = NULL;
  m_packet.size = 0;

  fflush(stdout);

  int got_output = 1;
  int ret = avcodec_encode_video2(m_context, &m_packet, m_picture, &got_output);

  if (ret < 0)
  {
    printf("Error encoding frame %3d\n", m_framecount);
  }

  if (got_output)
  {
    printf("writing frame %3d (size=%5d)\n", m_framecount, m_packet.size);
    fwrite(m_packet.data, 1, m_packet.size, m_file);
    av_packet_unref(&m_packet);
  }

  m_framecount++;
}

void ffmpeg_encoder::staticInit()
{
  if (!isStaticInit)
  {
    /*must be called before using avcodec lib */
    // deprecated avcodec_init();
	
    /* register all the codecs (you can also register only the codec you wish to have smaller code */
    avcodec_register_all();

    isStaticInit = true;
  }
}

