#ifndef THRESWINDOW_CXX
#define THRESWINDOW_CXX

#include "ThresWindow.h"
#include <numeric>

namespace opana
{

	const PulseVector_t ThresWindow::Reconstruct(const unsigned int ch,
					const Waveform_t& wf)
	{
		PulseVector_t ret;
		auto ped_info = _algo.Calculate(wf,1);

		float base_mean=ped_info.first;
		float base_std=ped_info.second;

		float threshold=base_mean+_threshold_factor*base_std;

		int n = wf.size();	
		int lastcross=0;
		opana::Pulse_t pulse;
		bool above=false;
		bool goodgap=false;
		bool goodwidth=false;
		std::vector<unsigned short>::iterator max_it;
	
	

		for (int samp=0; samp<n; ++samp)
		{

			if ((above==false)&&(wf[samp]>=threshold))
			{
				lastcross=samp;	
				above=true;
				if (samp>=(lastcross+_min_gap)) 
				{
					goodgap=true;
					pulse.tstart=samp;
				}
				else goodgap=false;
			}

			if ((above==true)&&(wf[samp]<=threshold))
			{
				lastcross=samp;	
				above=true;
				if (samp>=(lastcross+_min_width)) 
				{
					goodwidth=true;
					pulse.tend=samp;
				}
				else goodwidth=false;
			}

			if(goodwidth&&goodgap)
			{
				pulse.ch=ch;
				
				max_it=std::max_element(wf.begin()+pulse.tstart,wf.begin()+pulse.tend);
				pulse.tmax=std::distance(wf.begin(),max_it);
				pulse.amp=*max_it-base_mean;
				pulse.ped_mean = base_mean;
				pulse.reject_edge=reject_edges(wf);
				pulse.area=std::accumulate(wf.begin()+pulse.tstart,wf.end()+pulse.tend,0);
				pulse.area-=base_mean*(float)(pulse.tend-pulse.tstart);

				ret.push_back(pulse);

				goodwidth=false;
				goodgap=false;
			}
		}

		return ret;
	}
		
	
	float ThresWindow::vector_mean(const Waveform_t & wf,int start,int end) const
	{
		float sum = (float)std::accumulate(wf.begin()+start,wf.begin()+end,0);
		int n = wf.size(); 
		return sum/(float)n;
	}


	float ThresWindow::vector_std(const Waveform_t & wf,int start,int end) const
	{
		float sq_sum =  (float)std::inner_product(wf.begin()+start,wf.begin()+end,wf.begin()+start,0); 
		int n = wf.size();
		float mean = vector_mean(wf,start,end);

		return sq_sum/(float)n - mean*mean;
	}
	

	bool ThresWindow::reject_edges(const Waveform_t & wf) const
	{
		float start_rms = vector_std(wf,0,_baseline_samples);
		
		if (start_rms>_noise_factor*_noise_rms) return true;
		
		int n = wf.size()
		float end_rms = vector_std(wf,n-_baseline_samples,n);
	
		if (end_rms>_noise_factor*_noise_rms) return true;	

		return false;
	}


	void ThresWindow::Configure(float noise_rms, float noise_factor, float threshold_factor, int baseline_samples, int min_width, int min_gap)
	{
		_noise_rms = noise_rms;
		_noise_factor = noise_factor;
		_threshold_factor = threshold_factor;
		_baseline_samples = baseline_samples;
		_min_width = min_width;
		_min_gap = min_gap;
	}	


#endif
