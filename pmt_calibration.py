#! /usr/bin/python
import csv
import math
import os
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import ROOT
import sys
import scipy
import array
plt.style.use('ggplot')

filedir = "/home/zander/physics/tallbo_summer_2014/rundata/ledfinal"


nbaseline=50
min_peak_width=5 #samples
fixed_begin=0
fixed_end=999

base_rms=0.000378192539751


file_limit=10000

def load_waveform(waveform_filename):
	time,voltage = np.loadtxt(waveform_filename,delimiter=',',usecols=(3,4),unpack=True)
	return time,voltage

def baseline_measurements(voltage):
	baseline_voltage = voltage[:nbaseline]
	baseline_mean = np.mean(baseline_voltage)
	baseline_rms = np.std(baseline_voltage,ddof=1)
	print "BASELINE:",baseline_mean,"\n"
	return baseline_mean,baseline_rms

def find_pulses(voltage,baseline_mean,baseline_rms):
	threshold = baseline_mean-5*baseline_rms
	below=False
	neg_crossing=[]
	pos_crossing=[]
	npeaks=0
	n=0
	lasttrip=0
	for sample in voltage:
		if n>=(lasttrip+min_peak_width):	
			if (below==False) & (sample <= threshold):
				neg_crossing.append(n)
				lasttrip=n
				below=True
			if (below==True) & (sample >= threshold):
				pos_crossing.append(n)
				lasttrip=n
				below=False
				npeaks+=1
		n+=1

	return npeaks,neg_crossing,pos_crossing

def find_peak_charge(voltage,time,neg_crossing,pos_crossing,npeaks,baseline_mean):
	load_resistance=50 #Ohms
	charge=0
	if npeaks==0:
		return -1

#-----------------#
#	if npeaks>=1:
#		npeaks=1
#-----------------#

	for peak in range(0,npeaks):
		begin=neg_crossing[peak]
		end=pos_crossing[peak]
		delta_samples = end-begin
		delta_time = time[1]-time[0]
		sum_voltage = np.sum(voltage[begin:end])

		charge+=((sum_voltage-baseline_mean*delta_samples)*delta_time)/load_resistance
	return charge

def find_peak_min_voltage(min_voltage,min_voltage_samp,neg_crossing,pos_crossing,npeaks,baseline_mean):
	if npeaks==0:
		return 0

#-----------------#
#	if npeaks>=1:
#		npeaks=1
#-----------------#

	for peak in range(0,npeaks):
		begin=neg_crossing[peak]
		end=pos_crossing[peak]
		if begin<=min_voltage_samp<=end:
			return min_voltage-baseline_mean
	return 0

def find_fixed_charge(voltage,time,baseline_mean):
	load_resistance=50 #Ohms
	delta_samples = fixed_end-fixed_begin
	delta_time = time[1]-time[0]
	sum_voltage = np.sum(voltage[fixed_begin:fixed_end])

	charge=((sum_voltage-baseline_mean*delta_samples)*delta_time)/load_resistance
	return charge

def find_fixed_voltage(voltage,baseline_mean):
	return np.min(voltage[fixed_begin:fixed_end])-baseline_mean

def find_total_charge(voltage,time,baseline_mean):
	load_resistance=50 #Ohms

	delta_time = time[1]-time[0]
	sum_voltage = np.sum(voltage)

	charge=((sum_voltage-baseline_mean*len(time))*delta_time)/load_resistance
			
	return charge

def reject_edges(voltage,noise_rms):
	left_edge=voltage[0:nbaseline]
	right_edge=voltage[-nbaseline:]
	if (np.std(left_edge)>2.0*noise_rms):
		return 1
	elif(np.std(right_edge)>2.0*noise_rms):
		return 1
	else:
		return 0


#--------------------------------------------------------------------------#
#Get user input
#script, filename, waveformname, outfilename = sys.argv
script, filename = sys.argv


rootfilename=filename+str('.root')
outfilename=filename+str('_analysis.root')

f = ROOT.TFile("test2.root", "recreate")
t = ROOT.TTree("pmt_analysis", "PMT ANALYSIS")
#Soon to be ROOT branches:
#---------------------------#
min_voltage=np.zeros(4, dtype=float)
samp_min_voltage=np.zeros(4, dtype=float)
time_min_voltage=np.zeros(4, dtype=float)

min_voltage_in_peak=np.zeros(4, dtype=float)

total_charge=np.zeros(4, dtype=float)
peak_charge=np.zeros(4, dtype=float)

fixed_charge=np.zeros(4, dtype=float)
fixed_min_voltage=np.zeros(4, dtype=float)

baseline_mean=np.zeros(4, dtype=float)
baseline_rms=np.zeros(4, dtype=float)

firstpeak_start_samp=np.zeros(4, dtype=float)
firstpeak_start_time=np.zeros(4, dtype=float)
firstpeak_stop_samp=np.zeros(4, dtype=float)
firstpeak_stop_time=np.zeros(4, dtype=float)
edge_reject=np.zeros(4, dtype=float)

num_peaks=np.zeros(4, dtype=float)

n_samples=np.zeros(4, dtype=float)

t.Branch('minimum_voltage', min_voltage, 'minimum_voltage[4]/D')
t.Branch('minimum_voltage_sample', samp_min_voltage, 'minimum_voltage_sample[4]/D')
t.Branch('minimum_voltage_time', time_min_voltage, 'minimum_voltage_time[4]/D')
t.Branch('min_voltage_in_peak', min_voltage_in_peak, 'min_voltage_in_peak[4]/D')
t.Branch('fixed_charge', fixed_charge, 'fixed_charge[4]/D')
t.Branch('fixed_min_voltage', fixed_min_voltage, 'fixed_min_voltage[4]/D')
t.Branch('charge_in_peak', peak_charge, 'charge_in_peak[4]/D')
t.Branch('baseline_mean', baseline_mean, 'baseline_mean[4]/D')
t.Branch('baseline_rms', baseline_rms, 'baseline_rms[4]/D')
t.Branch('num_peaks', num_peaks, 'num_peaks[4]/D')
t.Branch('num_samples', n_samples, 'num_samples[4]/D')
t.Branch('firstpeak_tstart', firstpeak_start_time, 'firstpeak_tstart[4]/D')
t.Branch('firstpeak_nstart', firstpeak_start_samp, 'firstpeak_nstart[4]/D')
t.Branch('firstpeak_tstop', firstpeak_stop_time, 'firstpeak_tstop[4]/D')
t.Branch('firstpeak_nstop', firstpeak_stop_samp, 'firstpeak_nstop[4]/D')
t.Branch('edge_reject', edge_reject, 'edge_reject[4]/D')
#---------------------------#


file_counter=0



def np2py(numpy_array):
  _list=np.ndarray.tolist(numpy_array)
  return array.array('f',_list)


#Open ROOT waveform file
rootfile = ROOT.TFile(rootfilename,'READ')

datatree=rootfile.Get('waveformdata')
infotree=rootfile.Get('waveforminfo')



for entry in infotree:
    nchannels=entry.numchannels
    active_channels=entry.activechannels
    nsamp=entry.samples_per_waveform
    samptime=entry.secs_per_sample

used_channels=[]
for ch in range(0,4):
  if (active_channels[ch]==1):
    used_channels.append(ch)


print nchannels
print active_channels
print nsamp
print samptime
for st in samptime:
	print st
raw_input('')

channels=np.ndarray(shape=(4,nsamp))

count=0
for entry in datatree:
	print "NUM: ", count
	count+=1

	channels[0]=np.asarray(entry.ch1wfms)
	channels[1]=np.asarray(entry.ch2wfms)
	channels[2]=np.asarray(entry.ch3wfms)
	channels[3]=np.asarray(entry.ch4wfms)
	#channel.append(chy)
	#chx=np.arange(len(chy))
	#chx=np.multiply(chx,samptime[1])
	#channel.append(chx)

	for ch in range(0,4):
		#print samptime[ch]
		time=np.arange(0,nsamp)
		time=np.multiply(time,samptime[ch])
		#print len(time)
		#print len(channels[ch])
		if (ch==1):
				'''
				if (reject_edges(channels[ch],base_rms)==1):

					plt.plot(samps,channels[ch])
					plt.show()
					raw_input('')
				'''
				'''
				my_bsl_mean=np.mean(channels[ch][0:nbaseline])	
				if ((np.amin(channels[ch])-my_bsl_mean)<-5.0*base_rms):
					samps=np.arange(0,nsamp)	
					t1=np.linspace(0,nsamp,1000)
					l1=np.ones(len(t1))
					l1=np.multiply(l1,-5.0*base_rms)
					plt.plot(t1,l1,'b-')
					my_voltage=np.add(channels[ch],-1.0*my_bsl_mean)
				'''
				#samps=np.arange(0,nsamp)	
				#plt.plot(samps,channels[ch])
				#plt.show()
				#raw_input('')
		voltage=channels[ch]	
	
		n_samples[0]=len(time)
	
		#print "Voltages:"
		#print voltage
	
		baseline_mean[ch]=np.mean(channels[ch][0:nbaseline])
		baseline_rms[ch]=np.std(channels[ch][0:nbaseline])

		edge_reject[ch]=reject_edges(channels[ch],base_rms)


		'''
		if ((edge_reject[ch]==0)&(ch==1)&(np.amin(channels[ch])<-5.0*base_rms)):
			samps=np.arange(0,nsamp)  
			plt.plot(samps,channels[ch])
			plt.show()
			raw_input('')
		'''
		npulses,neg_crossing,pos_crossing = find_pulses(voltage,baseline_mean[ch],baseline_rms[ch])
		if((ch==1)&(npulses==1)&(reject_edges(channels[ch],base_rms)==0)):
			'''
			samps=np.arange(0,nsamp)
			plt.plot(samps,channels[ch])	
			t1=np.linspace(0,nsamp,1000)
			l1=np.ones(len(t1))
			l1=np.multiply(l1,-5.0*baseline_rms[ch]+baseline_mean[ch])
			plt.plot(t1,l1,'b-')
			for n in range(0,npulses):
				xval=np.zeros(1)
				yval=np.zeros(1)
				xval[0]=neg_crossing[n]
				yval[0]=channels[ch][neg_crossing[n]]
				plt.plot(xval,yval,'bo')
				xval[0]=pos_crossing[n]
				yval[0]=channels[ch][pos_crossing[n]]
				plt.plot(xval,yval,'go')
			plt.show()
			raw_input('')
			'''
		num_peaks[ch]=(npulses)

		if npulses>0:
			#print "Crossing:",neg_crossing[0];
			firstpeak_start_samp[ch]=neg_crossing[0];
			firstpeak_start_time[ch]=time[neg_crossing[0]];
			firstpeak_stop_samp[ch]=pos_crossing[0];
			firstpeak_stop_time[ch]=time[pos_crossing[0]];
		else:
			firstpeak_start_samp[ch]=len(time);
			firstpeak_start_time[ch]=time[len(time)-1];
			firstpeak_stop_samp[ch]=len(time);
			firstpeak_stop_time[ch]=time[len(time)-1];
	
			
	
	
		wfmpeak_charge = find_peak_charge(voltage,time,neg_crossing,pos_crossing,npulses,baseline_mean[ch])
		peak_charge[ch]=(wfmpeak_charge)
	
		wfmtotal_charge = find_total_charge(voltage,time,baseline_mean[ch])
		total_charge[ch]=(wfmtotal_charge)
		#print total_charge


		fixed_charge[ch] = find_fixed_charge(voltage,time,baseline_mean[ch]);
		'''
		if((ch==1)& (fixed_charge[ch]>-4.0e-12)):
			plt.plot(time,channels[ch])
			plt.show()
			raw_input('')
		'''
		wfmsamp_min_voltage = np.argmin(voltage)
		samp_min_voltage[ch]=(wfmsamp_min_voltage)
	
		wfmmin_voltage = voltage[wfmsamp_min_voltage]-baseline_mean[ch]
		min_voltage[ch]=(wfmmin_voltage)
		#print "min_voltage: ", min_voltage[ch], " Channel: ", ch
	
		wfmtime_min_voltage = time[wfmsamp_min_voltage]
		time_min_voltage[ch]=(wfmtime_min_voltage)
	
		#min_voltage_in_peak[ch]=find_peak_min_voltage(wfmmin_voltage,wfmsamp_min_voltage,neg_crossing,pos_crossing,npulses,wfmbaseline_mean)
	
		fixed_min_voltage[ch]=find_fixed_voltage(voltage,baseline_mean[ch])
		#print "VOLT: ",find_fixed_voltage(voltage,wfmbaseline_mean)
		'''
		if((ch==1)& (fixed_min_voltage[ch]<-0.3)):
			plt.plot(time,channels[ch])
			plt.show()
			raw_input('')
		'''

			

	
	t.Fill()
	
f.Write()
f.Close()



