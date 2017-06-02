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
#include <vector>

int main()
{
  int result = UnitTest::RunAllTests();;

  std::vector<int> v;

  auto l = [&]() {
    v.push_back(5);
  };

  l();

#if WIN32
  system("pause");
#endif

  return result;
}
