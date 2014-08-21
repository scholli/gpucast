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

#include <unittest++/UnitTest++.h>
#include <iostream>


int main()
{
  int result = UnitTest::RunAllTests();;

#if WIN32
  system("pause");
#endif

  return result;
}
