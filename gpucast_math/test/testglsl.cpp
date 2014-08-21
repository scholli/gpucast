/********************************************************************************
*
* Copyright (C) 2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testglsl.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <vector>
#include <gpucast/math/matrix.hpp>
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/glsl.hpp>

using namespace gpucast::math;

#define MAX_UINT 4294967295

///////////////////////////////////////////////////////////////////////////////
bool get_next_fragment ( std::vector<uvec4> const&  indexlist,
                         uint                       current_index,
                         uvec4                      current_fragment,
                         uint                       nfragments,
                         uvec4&                     next_fragment )
{
  int   index = int(current_index);
  bool  found = false;
  uint  depth = MAX_UINT;

  for ( unsigned i = 0; i != nfragments; ++i )
  {
    // get entry for this fragment
    uvec4 fragment = imageLoad ( indexlist, index );

    if ( depth               >  fragment[3] &&
         current_fragment[3] <  fragment[3] &&
         current_fragment[1] != fragment[1] )
    {
      next_fragment = fragment;
      depth         = fragment[3];
      found         = true;
    }

    // set index to next fragment
    index = int(fragment[0]);
  }

  return found;
}


///////////////////////////////////////////////////////////////////////////////
void close_fragment_list ( std::vector<uvec4>& indexlist,
                           int                 start_index,
                           uint                nfragments )
{
  int index = start_index;
  for ( unsigned i = 0; i != nfragments; ++i )
  {
    uvec4 entry = imageLoad ( indexlist, index );

    if ( i == nfragments - 1)
    {
      imageStore(indexlist, index, uvec4(start_index, entry[1], entry[2], entry[3]));
    }

    index = int(entry[0]);
  }
}


SUITE (matrix)
{

  TEST(types)
  {
    vec4 p0 = vec4(0.0, 1.0, 3.0, 0.0);
    vec3 p1 = vec3(2.0, 4.0, 0.0);
    vec2 p2 = vec2(5.0, 0.0);
    vec4 p3 = vec4(1.0f);

    CHECK (p3[0] == 1.0 && p3[1] == 1.0 && p3[2] == 1.0 && p3[3] == 1.0 );
    CHECK (p0[0] == 0.0 && p0[1] == 1.0 && p0[2] == 3.0 && p0[3] == 0.0 );
    CHECK (p1[0] == 2.0 && p1[1] == 4.0 && p1[2] == 0.0 );
    CHECK (p2[0] == 5.0 && p2[1] == 0.0 );
  }

  TEST(imageload)
  {
    std::vector<vec4> tmp (5, vec4(1.2f, 1.3f, 1.6f, 1.2f));
    vec4 p3 = imageLoad(tmp, 3);
    CHECK (p3 == vec4(1.2f, 1.3f, 1.6f, 1.2f));
  }

  TEST(fun0)
  {
    std::vector<uvec4> tmp (24);

    tmp[4] = uvec4(5,  0, 0, 56);
    tmp[5] = uvec4(6,  1, 0, 34);
    tmp[6] = uvec4(7,  2, 0, 53);
    tmp[7] = uvec4(16, 3, 0, 3);

    tmp[16] = uvec4(17, 4, 0, 67);
    tmp[17] = uvec4(18, 5, 0, 7);
    tmp[18] = uvec4(19, 6, 0, 35);
    tmp[19] = uvec4(36, 7, 0, 14);

    uvec4 current (0, 0, 0, 0);
    uvec4 next    (0, 0, 0, 0);

    close_fragment_list(tmp, 4, 8);

    //std::copy(tmp.begin(), tmp.end(), std::ostream_iterator<uvec4> (std::cout, "\n"));
  }

}
