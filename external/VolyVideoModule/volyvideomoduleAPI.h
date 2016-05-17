/*
VolyRenderer.

Authors: Alexander Bornik, Martin Urschler
Mail: medlib@lists.icg.tugraz.at, bornik@icg.tugraz.at, urschler@icg.tugraz.at

Copyright (C) 2010 Institute for Computer Graphics and Vision,
Graz University of Technology
&
Ludwig Boltzmann Institute for Clinical Forensic Imaging
Graz
*/


#ifndef VOLYVIDEOMODULEAPI_H
#define VOLYVIDEOMODULEAPI_H

#if defined(WIN32) || defined(_WIN32_WCE)
#  pragma warning(disable:4251)
#  pragma warning(disable:4290)
#  ifdef volyvideomodule_EXPORTS
#    define VOLYVIDEOMODULE_API __declspec(dllexport)
#  else
#    define VOLYVIDEOMODULE_API __declspec(dllimport)
#  endif
#else
#  define VOLYVIDEOMODULE_API
#endif

#endif // VOLYVIDEOMODULEAPI_H
