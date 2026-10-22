***************************************************************
** readme file associated with the manuscript: An investigation of how context affects the response of pedestrians to the movement of others**
***************************************************************

*All methods and experimental details are described in the manuscript entitled: "An investigation of how context affects the response of pedestrians to the movement of others" by Yunhe Tong and Nikolai W. F. Bode, published in Saftey Science.

*This dataset contains the information on participants and their route choice inside a three-dimensional virtual environment. Please refer to the associated publication for details on the study and data analysis.

*This data is fully anonymous and cannot be related to particular individuals.

*All data collected and prepared by Yunhe Tong and Nikolai W.F. Bode who should also be contacted with questions (yunhe.tong@bristol.ac.uk, nikolai.bode@bristol.ac.uk).

*A blank copy of the consent form for this experiment is included in this folder and named 'consent_form.pdf'.

*A screenshot of the information sheet that was shown on a monitor when participants signed the consent form is also included in this folder ('information_sheet.pdf').

*Ethical approval for this experiment was obtained from the Engineering Faculty Research Ethics Committee at the University of Bristol.

*Two types of data are included in this dataset:
(1) the raw data output from the interactive computer-based task for 600 human participants
(2) processed data for 547 participants that were used for Data analysis (see associated manuscript)

The presentation of this data is described in more detail below.

Throughout, missing data (e.g. age, gender) is denoted by 'NA'.


****************
(1) Raw data

*This data can be found in the folder 'raw_data'

*The data consists of 600 different folders,  each named by a unique ID for identifying IP addresses.  

* In the subfolder, the files show the data for participants in this specific treatment.
* Each participant's data is saved in a single file and the file names give the unique ID assigned by the online recruitment platform.  

* In the data file, the first 6 rows give information about the participants and the scenario. The remaining rows indicate the movements of the participant.

*Row 1 indicates the time step each participant started the experiment. 

*Row 2 indicates the spatial knowledge of participants.
	 0: the participant was provided with no spatial knowledge
	 1: the participant was provided with a map indicating the exit chosen by more simulated pedestrians was closer to the final exit
	 2: the participant was provided with a map indicating the exit chosen by less simulated pedestrians was closer to the final exit
	 3: the participant was provided a map indicating both exits were equidistant from the final exit

*Rows 3 and 4 indicate the number of simulated pedestrians evacuating through the left and right exits, respectively.

*Row 5 indicates the time step each participant finished the experiment. 

*Row 6 indicates the exit choice of participants: 1 for the left exit and 2 for the right exit.

*The remaining rows indicate the movements of the participant.
	*Each row presents data from one simulated time-step for the training section of the experiment in the virtual environment.
	*Columns 1 to 3  give the x, y and z coordinates for the positions of this human participant inside the virtual environment, respectively.
	*Column 4  lists the time step number from the start to the end of training.

****************

(2) Processed data

*This data can be found in the folder 'processed_data'

*This data consists of one plain text file called 'processed_data.csv'.

*This file contains processed data for all 547 human participants. 

*The age and gender of participants are obtained via their unique ID assigned by the online requirement platform and are included in 'processed_data.csv'. 

*The file has 8 columns and the first row provides short descriptive column names.

*Data for human participants are given in the remaining rows. All data for one participant is given in consecutive rows. 

*Columns 1 to 2 indicate the number of simulated pedestrians in the left exit (column 1) and right exit (column 2).

*Column 3 indicates the spatial knowledge of participants, same as the data arrangement in the raw data file. 

*Column 4 indicates the exit choice of participants: 1 for the left exit and 2 for the right exit.

*Columns 5 to 6 indicate the time each participant started (column 4) and finished (column 5) the experiment. 

*Column 7 indicates the gender of the participant. 0 for male, 1 for female, 2 for no answers and 3 for prefer not to say. 
	
*Column 8 indicates the age of the participant.

****************

** The virtual environment **

*see also figure 1 in the associated manuscript.


****************