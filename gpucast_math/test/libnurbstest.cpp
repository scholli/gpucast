/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/tmltest.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#if WIN32
  #include <UnitTest++.h>
#else
  #include <unittest++/UnitTest++.h>
#endif

#include <iostream>


int main()
{
  int result = UnitTest::RunAllTests();;

#if WIN32
  system("pause");
#endif

  return result;
}
