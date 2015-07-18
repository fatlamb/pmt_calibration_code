/**
 * \file ThresWindow.h
 *
 * \ingroup OpAnaTreeMaker
 * 
 * \brief Class def header for a class ThresWindow
 *
 * @author kazuhiro
 */

/** \addtogroup OpAnaTreeMaker

    @{*/
#ifndef THRESWINDOW_H
#define THRESWINDOW_H

#include <iostream>
#include <iterator>
#include <algorithm>
#include "PulseFinderBase.h"
#include "PedEstimator.h"
namespace opana {
  /**
     \class ThresWindow
     User defined class ThresWindow ... these comments are used to generate
     doxygen documentation!
  */
  class ThresWindow : public PulseFinderBase {
    
  public:
    
    /// Default constructor
    ThresWindow(){}
    
    /// Default destructor
    ~ThresWindow(){}

    const PulseVector_t Reconstruct(const unsigned int ch,
					    const Waveform_t& wf);

		/// Access to pedestal estimator algorithm
		PedEstimator& Algo() {return _algo;}

		/// Configure function
		void Configure(float noise_rms=0.000378192539751, float noise_factor=2.0, float threshold_factor=5.0, int baseline_samples=50, int min_width=5, int min_gap=5);

		float vector_mean(const Waveform_t& wf, int start, int end) const;
		float vector_std(const Waveform_t& wf, int start, int end) const;
		float reject_edges(const Waveform_t& wf) const;




	
	protected:
		PedEstimator _algo;

		float _noise_rms;
		float _noise_factor;
		float _threshold_factor;
		int _baseline_samples;
		int _min_width;
		int _min_gap;
  };
}

#endif
/** @} */ // end of doxygen group 

