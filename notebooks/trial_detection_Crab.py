import batanalysis as ba
import os
import numpy as np
#from heasoftpy import HeasoftpyExecutionError


#Read in the file, then create a bat survey object for all the observation

Crab_dir="/Users/slaha/Desktop/BAT_projects/GIT_Repository_directory_Jan2022/survey_data/Crab_data/"  #This is the path to the directory where your observations are kept

list_of_obsid=np.genfromtxt((Crab_dir+"list_of_obsid.txt"), usecols=(0),unpack=True, dtype=str) #For multiple observations need to create a text file of obsids.



batsurvey_obs=[]        #We are saving the BatSurvey objects in this list while we loop over.


input_dict=dict(indir=None,outdir=None,incatalog="survey6b_2.cat")       #Creating a dictionary to set up the input and output directories and the files. Note that you need to ensure that your source is in the catalog. Or else you may need to add it.


for i in list_of_obsid:

    print(i)
    try:
        
        obs=ba.BatSurvey(str(i),input_dict=input_dict,obs_dir=Crab_dir,recalc=True)
    
        obs.save()
        batsurvey_obs.append(obs)
    except ValueError:
        print("Obsid has no survey data")

'''
This portion of the code can be uncommented when you have already run the batsurvey observation once (above), and just need to use them to analyze the data. In that situation, you need to comment out the above for loop entirely.

batsurvey_obs=[]
for i in list_of_obsid:

    try:
        batsurvey_obs.append(ba.BatSurvey(str(i), load_file=os.path.join(Crab_dir,"%s_surveyresult/batsurvey.pickle"%(str(i)))))
        print("Loading observation id", i)
    except FileNotFoundError:
        pass

print("I am ending the loading process here")

'''


for obs in batsurvey_obs:

    print("Running calculate_PHA for observation id", obs.pointing_ids)
    obs.merge_pointings()
    obs.pointing_ids 
    obs.calculate_pha(id_list="Crab",clean_dir=True)
    ba.calc_response(obs.pha_file_names_list)

    fluxarray=[]
    fluxarray_lolim=[]
    fluxarray_hilim=[]

    for pha,point_id in zip(obs.pha_file_names_list,obs.pointing_ids): #Looping over individual PHA/pointings

        try:							#If XSPEC cannot fit because of negative counts
            flux,fluxerr,_=ba.fit_spectrum(pha,plot_fit=True)
        
            fluxarray.append(flux)
            fluxarray_lolim.append(fluxerr[0]) 
            fluxarray_hilim.append(fluxerr[1]) 

        except Exception as Error_with_Xspec_fitting:
            print (Error_with_Xspec_fitting) 
            fluxarray.append(0)
            fluxarray_lolim.append(0) 
            fluxarray_hilim.append(0) 

        
    ba.calculate_detection(obs,fluxarray,fluxarray_lolim,fluxarray_hilim,source_list="FRB180916",plot_fit=True)


ba.print_parameters(batsurvey_obs,values=["met_time","utc_time", "exposure","flux","flux_lolim", "flux_hilim"],savetable=True, save_file=Crab_dir+"200_onwards_output.txt", latex_table=True)

#to load all the saved data
#obs=[]
#for i in list_of_obsid:
#    obs.append(ba.BatSurvey(str(i), load_file=os.path.join(FRB180916_dir,"%s_surveyresult/batsurvey.pickle"%(str(i)))))

#to combine all of the light curves
#ba.combine_survey_lc(batsurvey_obs)

#to plot the light curve
#ba.plot_survey_lc(os.path.join(os.path.split(batsurvey_obs[0].result_dir)[0], "total_lc"), time_unit="MJD",id_list="FRB180916")
