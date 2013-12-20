/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : volume_view.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#pragma warning(disable: 4127) // Qt conditional expression is constant

// system includes
#include <GL/glew.h>
#include <QtGui/QApplication>

#include <mainwindow.hpp>
#include <glpp/error.hpp>

class volume_viewer : public QApplication
{
public :
  volume_viewer ( int argc, char** argv ) : QApplication(argc, argv ) {}
  virtual ~volume_viewer() {}
#if 1
  /* virtual */ bool notify ( QObject * receiver, QEvent * e )
  {
    bool result = false;
    try {
      result = QApplication::notify(receiver, e);
    } catch  ( std::exception& e ) {
      std::cerr << e.what() << std::endl;
    }
    return result;
  }
#endif
};


struct uint4 
{
  uint4()
   : x (0),
     y (0),
     z (0),
     w (0)
  {};

  uint4(unsigned a, unsigned b, unsigned c, unsigned d)
    : x (a),
      y (b),
      z (c),
      w (d)
  {}

  void print ( std::ostream& os ) const
  {
    os << "[ " << x << ", " << y << ", " << z << ", " << w << " ]";
  }

  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
};

#define int_t int
#define uint4_t uint4
#define __device__ 

inline int intBitsToFloat(unsigned i) { return i; }

///////////////////////////////////////////////////////////////////////////////
inline
void bubble_sort_indexlist_1 ( int      start_index,
                               unsigned nfragments,
                               uint4*   indexlist )
{
  // sort list of fragments
  for ( int i = 0; i != nfragments; ++i )
  { 
    int index   = start_index;

    for ( int j = 0; j != int(nfragments-1); ++j )
    { 
      // get entry for this fragment
      uint4 entry0 = indexlist[index];

      if ( entry0.x == 0 ) {
        break;
      } else {
        uint4 entry1 = indexlist[int_t(entry0.x)];

        if ( intBitsToFloat(int_t(entry0.w)) > intBitsToFloat(int_t(entry1.w)) ) 
        {
          // swizzle depth and related data
          indexlist[index          ] = uint4_t(entry0.x, entry1.y, entry1.z, entry1.w); 
          indexlist[int_t(entry0.x)] = uint4_t(entry1.x, entry0.y, entry0.z, entry0.w);
        }

        index = int_t(entry0.x);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
bool get_next_fragment ( uint4 const& fragment,
                         uint4 const* indexlist,
                         int&         next_index,
                         uint4&       next_fragment ) 
{
  next_fragment = fragment;

  while ( next_fragment.x != 0 )
  {
    next_index    = next_fragment.x;
    next_fragment = indexlist[next_index];

    if ( next_fragment.z == fragment.z )
    {
      return true;
    } 
  }

  return false;
}
                         
///////////////////////////////////////////////////////////////////////////////
__device__ inline
void shift_fragments ( int    first_index,
                       int    source_index,
                       uint4* indexlist )
{
  uint4 source             = indexlist[source_index];

  int   backup_index       = first_index;
  uint4 backup_fragment    = indexlist[backup_index];

  indexlist[first_index]   = uint4_t(backup_fragment.x, source.y, source.z, source.w);

  while ( backup_fragment.x != source_index ) 
  {
    uint4 tmp = indexlist[backup_fragment.x];
    indexlist[backup_fragment.x] = uint4_t(tmp.x, backup_fragment.y, backup_fragment.z, backup_fragment.w);
    backup_fragment = tmp;
  }

  indexlist[source_index] = uint4_t(source.x, backup_fragment.y, backup_fragment.z, backup_fragment.w);
}

void print (  uint4*   indexlist, int size )
{
  for ( int i = 0; i < size; ++i )
  {
    std::cout << i << " : "; indexlist[i].print(std::cout); std::cout << std::endl;
  }
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void merge_intervals ( int      start_index,
                       uint4*   indexlist )
{
  int   index    = start_index;
  uint4 fragment = indexlist[index];

  int   next_index = 0;
  uint4 next_fragment;

  while ( true )
  {
    if ( !get_next_fragment ( fragment, indexlist, next_index, next_fragment ) ) {
      break;
    }
  
    if ( next_index != fragment.x )
    {
      shift_fragments(fragment.x, next_index, indexlist);
    } 

    next_fragment = indexlist[fragment.x];

    if ( next_fragment.x != 0 )
    {
      index    = next_fragment.x;
      fragment = indexlist[index];
    } else {
      break;
    }
  }
}




int main(int argc, char **argv)
{
#if 1
  try 
  {
    volume_viewer app(argc, argv);
    if ( argc == 3 ) 
    {
      mainwindow win(argc, argv, std::atoi(argv[1]), std::atoi(argv[2]));
      app.exec();
    } else {
      mainwindow win(argc, argv, 1024, 1024);
      app.exec();
    }
  } 
  catch ( std::exception& e ) 
  {
    std::cerr << e.what() << std::endl;
    system("pause");
    return 0;
  } 
#else

  std::vector<uint4> indexlist(4);
  indexlist.push_back( uint4(5,  0, 5, 5) );
  indexlist.push_back( uint4(6,  0, 5, 8) );
  indexlist.push_back( uint4(7,  0, 3, 3) );
  indexlist.push_back( uint4(8,  0, 4, 4) );
  indexlist.push_back( uint4(9,  0, 3, 1) );
  indexlist.push_back( uint4(10, 0, 6, 6) );
  indexlist.push_back( uint4(11, 0, 6, 7) );
  indexlist.push_back( uint4(0,  0, 4, 2) );

  std::for_each(indexlist.begin(), indexlist.end(), [] (uint4 const& a) { a.print(std::cout); std::cout << std::endl; } );

  bubble_sort_indexlist_1 ( 4, 8, &indexlist[0]);

  std::for_each(indexlist.begin(), indexlist.end(), [] (uint4 const& a) { a.print(std::cout); std::cout << std::endl; } );

  merge_intervals ( 4, &indexlist[0]);

  std::for_each(indexlist.begin(), indexlist.end(), [] (uint4 const& a) { a.print(std::cout); std::cout << std::endl; } );

#endif
}
