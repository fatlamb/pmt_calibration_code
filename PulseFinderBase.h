/**
 * \file PulseFinderBase.h
 *
 * \ingroup OpAnaTreeMaker
 * 
 * \brief Class def header for a class PulseFinderBase
 *
 * @author kazuhiro
 */

/** \addtogroup OpAnaTreeMaker

    @{*/
#ifndef PULSEFINDERBASE_H
#define PULSEFINDERBASE_H

#include <iostream>
#include <vector>
namespace opana {

  struct Pulse_t {
    short ch; //channel
    short tstart;
    short tend;
    short tmax;
    float ped_mean;
    float amp;
    float area;
		bool reject_edge;

    Pulse_t() {
      ch = -1;
      tstart = tend = tmax = -1;
      ped_mean = -1;
      amp = area = -1;
    }

    void dump() {
      std::cout  << "\n\t==start=="
		 << "\n\tch:       " << ch
		 << "\n\ttstart:   " << tstart
		 << "\n\ttend:     " << tend
		 << "\n\ttmax:     " << tmax
		 << "\n\tped_mean: " << ped_mean
		 << "\n\tamp:      " << amp
		 << "\n\tarea:     " << area
		 << "\n\t==end==\n";
	
    }
  };

  typedef std::vector<unsigned short> Waveform_t;
  typedef std::vector<opana::Pulse_t> PulseVector_t;
  
  /**
     \class PulseFinderBase
     User defined class PulseFinderBase ... these comments are used to generate
     doxygen documentation!
  */
  class PulseFinderBase{
    
  public:
    
    /// Default constructor
    PulseFinderBase(){}
    
    /// Default destructor
    virtual ~PulseFinderBase(){}
    
    virtual const PulseVector_t Reconstruct(const unsigned int ch,
					    const Waveform_t& wf) = 0;
    
  };
}
#endif
/** @} */ // end of doxygen group 

