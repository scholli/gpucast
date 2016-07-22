/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : igs_loader.cpp
*  project    : gpucast
*  description: part of inventor fileloader
*
********************************************************************************/

// header i/f
#include "gpucast/core/import/igs_loader.hpp"

// header, system
#include <fstream>

// header, project
#include <gpucast/core/import/igs_grammar.hpp>
#include <gpucast/core/import/igs_actor.hpp>
#include <gpucast/core/import/igs_stack.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>


namespace gpucast {

  // helper functor
  class spacer {
  public :
    spacer(std::vector<char>& v);
    char operator()(char const& c);

  private :
    std::vector<char>& v_;
    std::size_t i_;
  };

  spacer::spacer(std::vector<char>& v)
    : v_(v),
      i_(0)
  {}

  char spacer::operator()(char const& c)
  {
    if (i_ == 72) {
      v_.insert(v_.end() - 72, c);
      v_.insert(v_.end() - 72, ' ');

      v_.push_back(' ');
      v_.push_back('^');
      v_.push_back(' ');

      switch (c) {
      case 'D' :

        // separate columns
        for (int i = 67;i > 0; i-=8) {
	  v_.insert(v_.end() - i, ' ');
        }
        break;
      case 'P' :
        // separate last col
        v_.insert(v_.end() - 10, ' ');
        v_.insert(v_.end() - 10, ' ');
        v_.insert(v_.end() - 10, '^');
        v_.insert(v_.end() - 10, '^');
        break;
      default :
        break;
        // do nothing
      }
    }

    // secure space
    if (i_ == 73){
      v_.push_back(' ');
    }

    if (c == '\n') {
      i_ = 0;
    } else {
	    ++i_;
    }
    return c;
  }

  ////////////////////////////////////////////////////////////////////////////////
  std::vector<std::shared_ptr<nurbssurfaceobject>>
  igs_loader::load( std::string const& file, bool normalize )
  {
    _result = std::make_shared<nurbssurfaceobject>();

    // try to open file
    std::fstream fstr;
    fstr.open(file.c_str(), std::ios::in);

    // if file good parse stream
    if (fstr.good())
    {
      if (!_load(fstr))
      {
        _result.reset();
      }
    } else {
      _error = "igs_loader::load(): Could not open file " + file;
      _result.reset();
    }

    fstr.close();

    // normalize 
    if (normalize) {
      for (std::vector<nurbssurface>::iterator s = _result->begin(); s != _result->end(); ++s) {
        s->normalize();
      }
    }

    return{ _result };
  }

  ////////////////////////////////////////////////////////////////////////////////
  std::string         
  igs_loader::error_message() const
  {
    return _error;
  }

  ////////////////////////////////////////////////////////////////////////////////
  bool
  igs_loader::_load( std::fstream& istr )
  {
    istr.unsetf(std::ios::skipws);

    std::vector<char> tmp;
    std::transform(std::istream_iterator<char>(istr),
		               std::istream_iterator<char>(),
		               std::back_inserter(tmp),
		               spacer(tmp));

    // instance grammar and parse stream
    boost::spirit::classic::parse_info<std::vector<char>::iterator> info;
    info = boost::spirit::classic::parse(tmp.begin(), tmp.end(), igs_grammar(_result));

    _error = std::string(info.stop, tmp.end());

    return (_error.length() > 1) ? false : true;
  }

} // namespace gpucast
