import numpy as np
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import time

# Output file for results
of = open("output.txt", "w", encoding='ascii')

# Set seeds for consistent results each run
import random as rand
import tensorflow as tf
seed_value = 42
np.random.seed(seed_value)
rand.seed(seed_value)
tf.random.set_seed(seed_value)

# Loading data for the English League
with open('events_England.json', encoding='utf-8') as f:
    data_England = json.load(f)

# Converting the laoded into datframe
event_England = pd.DataFrame(data_England)

# Loading data for the French League
with open('events_France.json', encoding='utf-8') as f:
    data_France = json.load(f)

# Converting the laoded into datframe
event_France = pd.DataFrame(data_France)

# Loading data for Italian League
with open('events_Italy.json', encoding='utf-8') as f:
    data_Italy = json.load(f)

# Converting the laoded into datframe
event_Italy = pd.DataFrame(data_Italy)

# Loading data for Spanish league
with open('events_Spain.json', encoding='utf-8') as f:
    data_Spain = json.load(f)

# Converting the laoded into datframe
event_Spain = pd.DataFrame(data_Spain)

# Loading data for the German league
with open('events_Germany.json', encoding='utf-8') as f:
    data_Germany = json.load(f)

# Converting the laoded into datframe
event_Germany = pd.DataFrame(data_Germany)

# Combining all our dataframes to make one consolidated event data set for the 5 maor leagues
event_data = pd.concat([event_England, event_France, event_Germany, event_Italy, event_Spain])

# Reset the index of the resulting dataframe
event_data = event_data.reset_index(drop=True)

# Loading the player rank data
with open('playerank.json', encoding='utf-8') as f:
    data_player_rank = json.load(f)

# Converting the laoded into datframe
player_rank_df = pd.DataFrame(data_player_rank)


# Adding player rank and to the event_data dataset so that quality of the player taking the shot is included as a feature in our data. Joining the event_data and player rank data on the playerId column. To achieve this purpose we will be performing a <b>left</b> join between the two dataframes using <b>matchId</b> and <b>playerId</b> columns.
# 


event_data_merged = pd.merge(event_data, player_rank_df[['matchId','playerId','playerankScore']], 
                             on = ['matchId','playerId'], 
                             how = 'left')

# ### Filtering out Shots Data
# The goal is filter out all 'Shot' events excluding free kicks, penalty kicks and headers. For this, we apply a filter on the subevents to capture only 'Shot' events. But these events also contain headers. Headed shots can be identified using tags. Tag for headed shots is 403.

# Filtering out all the shots from our dataset
shots_df = event_data_merged[event_data_merged['subEventName']=='Shot'].reset_index(drop=True)

# ### Adding More Features to our Dataset
# We will be adding the following features to our data set
# 
# Tag         Description
# 
# 401         Left foot
# 
# 402         Right foot
# 
# 1201        Position: Goal low center
# 
# 1202        Position: Goal low right
# 
# 1203        Position: Goal center
# 
# 1204        Position: Goal center left
# 
# 1205        Position: Goal low left
# 
# 1206        Position: Goal center right
# 
# 1207        Position: Goal high center
# 
# 1208        Position: Goal high left
# 
# 1209        Position: Goal high right


left_foot = [] # Left-Footed Shots: 401
right_foot = [] # Right_Footed Shots: 402
glc = [] # Goal Low Center: 1201
glr = [] # Goal Low Right: 1202
gll = [] # Goal Low Left : 1205
gc = [] # Goal Center : 1203
gcl = [] # Goal Center Left: 1204
gcr = [] # Goal Center Right: 1206
ghc = [] # Goal High Center: 1207
ghr = [] # Goal High Right: 1209
ghl = [] # Goal High Left: 1208


row_number = 0
row_list = []
for tag in shots_df['tags']:
    for pair in tag:
        if pair['id'] == 403:
            row_list.append('Header_row')
        if pair['id'] != 403:
            if pair['id'] == 401:
                left_foot.append([row_number, 1])
            elif pair['id'] == 402:
                right_foot.append([row_number, 1])
            elif pair['id'] == 1201:
                glc.append([row_number, 1])
            elif pair['id'] == 1202:
                glr.append([row_number, 1])
            elif pair['id'] == 1203:
                gc.append([row_number, 1])
            elif pair['id'] == 1204:
                gcl.append([row_number, 1])
            elif pair['id'] == 1205:
                gll.append([row_number, 1])
            elif pair['id'] == 1206:
                gcr.append([row_number, 1])
            elif pair['id'] == 1207:
                ghc.append([row_number, 1])
            elif pair['id'] == 1208:
                ghl.append([row_number, 1])
            elif pair['id'] == 1209:
                ghr.append([row_number, 1])
                      
        
    row_number += 1

# Creating dataframes
left_foot_df = pd.DataFrame(left_foot)
left_foot_df.columns = ['index','Left_Foot']

right_foot_df = pd.DataFrame(right_foot)
right_foot_df.columns = ['index','Right_Foot']

glc_df = pd.DataFrame(glc)
glc_df.columns = ['index','GLC']

glr_df = pd.DataFrame(glr)
glr_df.columns = ['index','GLR']

gll_df = pd.DataFrame(gll)
gll_df.columns = ['index','GLL']

gc_df = pd.DataFrame(gc)
gc_df.columns = ['index','GC']

gcl_df = pd.DataFrame(gcl)
gcl_df.columns = ['index','GCL']

gcr_df = pd.DataFrame(gcr)
gcr_df.columns = ['index','GCR']

ghc_df = pd.DataFrame(ghc)
ghc_df.columns = ['index','GHC']

ghr_df = pd.DataFrame(ghr)
ghr_df.columns = ['index','GHR']

ghl_df = pd.DataFrame(ghl)
ghl_df.columns = ['index','GHL']

# concatenate all dataframes vertically
df_all = pd.concat([left_foot_df.set_index('index'), right_foot_df.set_index('index'), 
                    glc_df.set_index('index'), glr_df.set_index('index'), gll_df.set_index('index'), 
                    gc_df.set_index('index'), gcl_df.set_index('index'), gcr_df.set_index('index'), 
                    ghc_df.set_index('index'), ghr_df.set_index('index'), ghl_df.set_index('index')], axis=1)

# merge all dataframes based on index of df_A and index column of other dataframes
final_df = pd.merge(shots_df, df_all, left_index=True, right_index=True, how = 'left')

# fill all NaN values with 0
final_df = final_df.fillna(0)

# display the modified dataframe

# ## Features Selected for Model Building
# Following features will be selected will be added to the final model:
# 
#     1. X, Y Coordinates of Shot
#     
#     2. Angle of Shot from Goal
#     
#     3. Foot Used for Shooting (Left or Right Foot, Header are excluded from this study)
#     
#     4. Position of Shot in the Goal (Low left, Low right, Low center etc)
#     
#     5. Match Half in which the shot occured (1st half or 2nd half)
#     
# The base code for creation of our shots model is adopted from Friends of Tracking github page. Ammendments are made in the base code to add additional features described above:
# https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython/blob/master/3xGModel.py
# 
# Keep in mind the description of position column. The x and y coordinates are always in the range [0, 100] and indicate the percentage of the field from the perspective of the attacking team. In particular, the value of the x coordinate indicates the event's nearness (in percentage) to the opponent's goal, while the value of the y coordinates indicates the event's nearness (in percentage) to the right side of the field. So to get the actual X and Y coordinates, distance and angle from goal, we need certain transformations.

# Let us build the required dataset for expected goal model building
# Creating your feature matrix and dropping non-essential columns
shots_model=pd.DataFrame(columns=['Goal','X','Y','Player_Rank', 'Match_Period',
                                  'Left_Foot', 'Right_Foot',
                                  'GLC','GLR','GLL','GC','GCR','GCL','GHC','GHR','GHL'])

#Go through the dataframe, calculate X, Y co-ordinates, angle and fetch the remaining features 

for i,shot in final_df.iterrows():
    header=0
    for shottags in shot['tags']:
        if shottags['id']==403:
            header=1
    #Only include non-headers        
    if not(header):        
        shots_model.at[i,'X']=100-shot['positions'][0]['x']
        shots_model.at[i,'Y']=shot['positions'][0]['y']
        shots_model.at[i,'C']=abs(shot['positions'][0]['y']-50)
    
        #Distance in metres and shot angle in radians.
        x=shots_model.at[i,'X']*105/100
        y=shots_model.at[i,'C']*65/100
        shots_model.at[i,'Distance']=np.sqrt(x**2 + y**2)
        a = np.arctan( 7.32 *x /(x**2 + y**2 - (7.32/2)**2))
        if a<0:
            a=np.pi+a
        shots_model.at[i,'Angle'] =a
    
        #Was it a goal
        shots_model.at[i,'Goal']=0
        for shottags in shot['tags']:
            #Tags contain that its a goal
            if shottags['id']==101:
                shots_model.at[i,'Goal']=1
        # Adding the player rank
        shots_model.at[i,'Player_Rank'] = shot['playerankScore']
        
        # Adding match half
        shots_model.at[i,'Match_Period'] = shot['matchPeriod']
        
        # Adding one hot encoded features related to foot used for shooting and position of shot w.r.t goal
        shots_model.at[i,'Left_Foot']  = shot['Left_Foot']
        shots_model.at[i,'Right_Foot'] = shot['Right_Foot']
        shots_model.at[i,'GLC']        = shot['GLC']
        shots_model.at[i,'GLR']        = shot['GLR']
        shots_model.at[i,'GLL']        = shot['GLL']
        shots_model.at[i,'GC']         = shot['GC']
        shots_model.at[i,'GCL']        = shot['GCL']
        shots_model.at[i,'GCR']        = shot['GCR']
        shots_model.at[i,'GHC']        = shot['GHC']
        shots_model.at[i,'GHL']        = shot['GHL']
        shots_model.at[i,'GHR']        = shot['GHR']
        
# Let's see our finalized data set

# The angle calulcated is in radians. Let us first convert all the angles in degrees
def rad_deg(ang):
    return np.rad2deg(ang)

# Calling our function to achieve conversion
shots_model['Angle'] = shots_model['Angle'].apply(rad_deg)

# Match_Period is a categorical variable. So we will perform one-hot encoding for the column

# perform one-hot encoding for Match_Period column
one_hot = pd.get_dummies(shots_model['Match_Period'])

#merge one-hot encoded columns back with original DataFrame
shots_model = pd.concat([shots_model, one_hot], axis=1)

#drop the original 'Match_Period' column
shots_model.drop('Match_Period', axis=1, inplace=True)

# Resetting the index of the dataframe
shots_model = shots_model.reset_index(drop=True)

# Finalized dataframe

# Checking data types

# Correcting the data types of columns
# Changing all the one-hot encoded columns into integers
# Changing data type of Player_Rank column to float

shots_model = shots_model.astype({'X': float, 'Y': float, 'Goal': int, 'Left_Foot': int, 'Right_Foot': int ,'GLC': int, 'GLR': int, 
                                  'GLL': int, 'GC': int, 'GCR': int, 'GCL': int, 'GHC': int, 'GHR': int, 
                                  'GHL' : int, '1H': int, '2H': int, 'Player_Rank': float})

# Again checking our datatypes

shots_model.isna().sum()

# # <b>Visualizing our Data</b>
# 
# ## Creating a Football Pitch
# Let us firt create a function to plot a football pitch in python. This code is taken from the following git hub opage which in turn have used the code provided by FCPython.
# https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython/blob/master/FCPython.py

from matplotlib.patches import Arc
def createPitch(length,width, unity,linecolor): # in meters
    
    """
    creates a plot in which the 'length' is the length of the pitch (goal to goal).
    And 'width' is the width of the pitch (sideline to sideline). 
    Fill in the unity in meters or in yards.
    """
    #Set unity
    if unity == "meters":
        # Set boundaries
        if length >= 120.5 or width >= 75.5:
            return(str("Field dimensions are too big for meters as unity, didn't you mean yards as unity?\
                       Otherwise the maximum length is 120 meters and the maximum width is 75 meters. Please try again"))
        #Run program if unity and boundaries are accepted
        else:
            #Create figure
            figP=plt.figure()
            #figP.set_size_inches(7, 5)
            axP=figP.add_subplot(1,1,1)
           
            #Pitch Outline & Centre Line
            plt.plot([0,0],[0,width], color=linecolor)
            plt.plot([0,length],[width,width], color=linecolor)
            plt.plot([length,length],[width,0], color=linecolor)
            plt.plot([length,0],[0,0], color=linecolor)
            plt.plot([length/2,length/2],[0,width], color=linecolor)
            
            #Left Penalty Area
            plt.plot([16.5 ,16.5],[(width/2 +16.5),(width/2-16.5)],color=linecolor)
            plt.plot([0,16.5],[(width/2 +16.5),(width/2 +16.5)],color=linecolor)
            plt.plot([16.5,0],[(width/2 -16.5),(width/2 -16.5)],color=linecolor)
            
            #Right Penalty Area
            plt.plot([(length-16.5),length],[(width/2 +16.5),(width/2 +16.5)],color=linecolor)
            plt.plot([(length-16.5), (length-16.5)],[(width/2 +16.5),(width/2-16.5)],color=linecolor)
            plt.plot([(length-16.5),length],[(width/2 -16.5),(width/2 -16.5)],color=linecolor)
            
            #Left 5-meters Box
            plt.plot([0,5.5],[(width/2+7.32/2+5.5),(width/2+7.32/2+5.5)],color=linecolor)
            plt.plot([5.5,5.5],[(width/2+7.32/2+5.5),(width/2-7.32/2-5.5)],color=linecolor)
            plt.plot([5.5,0.5],[(width/2-7.32/2-5.5),(width/2-7.32/2-5.5)],color=linecolor)
            
            #Right 5 -eters Box
            plt.plot([length,length-5.5],[(width/2+7.32/2+5.5),(width/2+7.32/2+5.5)],color=linecolor)
            plt.plot([length-5.5,length-5.5],[(width/2+7.32/2+5.5),width/2-7.32/2-5.5],color=linecolor)
            plt.plot([length-5.5,length],[width/2-7.32/2-5.5,width/2-7.32/2-5.5],color=linecolor)
            
            #Prepare Circles
            centreCircle = plt.Circle((length/2,width/2),9.15,color=linecolor,fill=False)
            centreSpot = plt.Circle((length/2,width/2),0.8,color=linecolor)
            leftPenSpot = plt.Circle((11,width/2),0.8,color=linecolor)
            rightPenSpot = plt.Circle((length-11,width/2),0.8,color=linecolor)
            
            #Draw Circles
            axP.add_patch(centreCircle)
            axP.add_patch(centreSpot)
            axP.add_patch(leftPenSpot)
            axP.add_patch(rightPenSpot)
            
            #Prepare Arcs
            leftArc = Arc((11,width/2),height=18.3,width=18.3,angle=0,theta1=308,theta2=52,color=linecolor)
            rightArc = Arc((length-11,width/2),height=18.3,width=18.3,angle=0,theta1=128,theta2=232,color=linecolor)
            
            #Draw Arcs
            axP.add_patch(leftArc)
            axP.add_patch(rightArc)
            #Axis titles

    #check unity again
    elif unity == "yards":
        #check boundaries again
        if length <= 95:
            return(str("Didn't you mean meters as unity?"))
        elif length >= 131 or width >= 101:
            return(str("Field dimensions are too big. Maximum length is 130, maximum width is 100"))
        #Run program if unity and boundaries are accepted
        else:
            #Create figure
            figP=plt.figure()
            #figP.set_size_inches(7, 5)
            axP=figP.add_subplot(1,1,1)
           
            #Pitch Outline & Centre Line
            plt.plot([0,0],[0,width], color=linecolor)
            plt.plot([0,length],[width,width], color=linecolor)
            plt.plot([length,length],[width,0], color=linecolor)
            plt.plot([length,0],[0,0], color=linecolor)
            plt.plot([length/2,length/2],[0,width], color=linecolor)
            
            #Left Penalty Area
            plt.plot([18 ,18],[(width/2 +18),(width/2-18)],color=linecolor)
            plt.plot([0,18],[(width/2 +18),(width/2 +18)],color=linecolor)
            plt.plot([18,0],[(width/2 -18),(width/2 -18)],color=linecolor)
            
            #Right Penalty Area
            plt.plot([(length-18),length],[(width/2 +18),(width/2 +18)],color=linecolor)
            plt.plot([(length-18), (length-18)],[(width/2 +18),(width/2-18)],color=linecolor)
            plt.plot([(length-18),length],[(width/2 -18),(width/2 -18)],color=linecolor)
            
            #Left 6-yard Box
            plt.plot([0,6],[(width/2+7.32/2+6),(width/2+7.32/2+6)],color=linecolor)
            plt.plot([6,6],[(width/2+7.32/2+6),(width/2-7.32/2-6)],color=linecolor)
            plt.plot([6,0],[(width/2-7.32/2-6),(width/2-7.32/2-6)],color=linecolor)
            
            #Right 6-yard Box
            plt.plot([length,length-6],[(width/2+7.32/2+6),(width/2+7.32/2+6)],color=linecolor)
            plt.plot([length-6,length-6],[(width/2+7.32/2+6),width/2-7.32/2-6],color=linecolor)
            plt.plot([length-6,length],[(width/2-7.32/2-6),width/2-7.32/2-6],color=linecolor)
            
            #Prepare Circles; 10 yards distance. penalty on 12 yards
            centreCircle = plt.Circle((length/2,width/2),10,color=linecolor,fill=False)
            centreSpot = plt.Circle((length/2,width/2),0.8,color=linecolor)
            leftPenSpot = plt.Circle((12,width/2),0.8,color=linecolor)
            rightPenSpot = plt.Circle((length-12,width/2),0.8,color=linecolor)
            
            #Draw Circles
            axP.add_patch(centreCircle)
            axP.add_patch(centreSpot)
            axP.add_patch(leftPenSpot)
            axP.add_patch(rightPenSpot)
            
            #Prepare Arcs
            leftArc = Arc((11,width/2),height=20,width=20,angle=0,theta1=312,theta2=48,color=linecolor)
            rightArc = Arc((length-11,width/2),height=20,width=20,angle=0,theta1=130,theta2=230,color=linecolor)
            
            #Draw Arcs
            axP.add_patch(leftArc)
            axP.add_patch(rightArc)
                
    #Tidy Axes
    plt.axis('off')
    
    return figP,axP


def createGoalMouth():
    #Create figure
    figGM=plt.figure()
    axGM=figGM.add_subplot(1,1,1)

    linecolor='black'

    #Pitch Outline & Centre Line
    plt.plot([0,65],[0,0], color=linecolor)
    plt.plot([65,65],[50,0], color=linecolor)
    plt.plot([0,0],[50,0], color=linecolor)
    
    #Left Penalty Area
    plt.plot([12.5,52.5],[16.5,16.5],color=linecolor)
    plt.plot([52.5,52.5],[16.5,0],color=linecolor)
    plt.plot([12.5,12.5],[0,16.5],color=linecolor)
    
    #Left 6-yard Box
    plt.plot([41.5,41.5],[5.5,0],color=linecolor)
    plt.plot([23.5,41.5],[5.5,5.5],color=linecolor)
    plt.plot([23.5,23.5],[0,5.5],color=linecolor)
    
    #Goal
    plt.plot([41.5-5.34,41.5-5.34],[-2,0],color=linecolor)
    plt.plot([23.5+5.34,41.5-5.34],[-2,-2],color=linecolor)
    plt.plot([23.5+5.34,23.5+5.34],[0,-2],color=linecolor)
    
    #Prepare Circles
    leftPenSpot = plt.Circle((65/2,11),0.8,color=linecolor)
    
    #Draw Circles
    axGM.add_patch(leftPenSpot)
    
    #Prepare Arcs
    leftArc = Arc((32.5,11),height=18.3,width=18.3,angle=0,theta1=38,theta2=142,color=linecolor)
    
    #Draw Arcs
    axGM.add_patch(leftArc)
    
    #Tidy Axes
    plt.axis('off')
    
    return figGM,axGM


# ## From Where Most Shots have Taken Place?


#Two dimensional histogram
H_Shot=np.histogram2d(shots_model['X'], shots_model['Y'],bins=50,range=[[0, 100],[0, 100]])
goals_only=shots_model[shots_model['Goal']==1]
H_Goal=np.histogram2d(goals_only['X'], goals_only['Y'],bins=50,range=[[0, 100],[0, 100]])

#Plot the number of shots from different points
(fig,ax) = createGoalMouth()
pos=ax.imshow(H_Shot[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.RdYlGn_r)
fig.colorbar(pos, ax=ax)
ax.set_title('Number of shots')
plt.xlim((-1,66))
plt.ylim((-3,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('num_shots.png')
plt.clf()

# We can see that majority of the shots occured from within the box around the penalty spot area. Another thing we can see is that shots from flanks are very rare which is quite understandable as flanks are mostly used providing crosses or passes. Next let us visualize all the goal scoring locations.
# 
# ## From Where Majority of Goals have Been Scored?

#Plot the number of GOALS from different points
(fig,ax) = createGoalMouth()
pos=ax.imshow(H_Goal[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.RdYlGn_r)
fig.colorbar(pos, ax=ax)
ax.set_title('Number of goals')
plt.xlim((-1,66))
plt.ylim((-3,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
#fig.savefig('NumberOfGoals.pdf', dpi=None, bbox_inches="tight") 
fig.savefig('NumberOfGoals.png') 
plt.clf()

# The visual shows that of all the shots that were taken, shots from within the inner box are the most lethal. As distance or angle from the goal incerases, number of goals also decreases with very few goals from outside the box. No wonder goals from outside the box are a rarity (and absolute bilnders as well!). Next we plot the probability of scoring from different poitns on the pitch.
# 
# ## Which Areas of the Pitch have the Highest Probability of Producing a Goal?

#Plot the probability of scoring from different points
(fig,ax) = createGoalMouth()
pos=ax.imshow(H_Goal[0]/H_Shot[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.RdYlGn_r,vmin=0, vmax=0.5)
fig.colorbar(pos, ax=ax)
ax.set_title('Proportion of shots resulting in a goal')
plt.xlim((-1,66))
plt.ylim((-3,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('prob_scoring.png')
plt.clf()


# At first sight the visual looks in order. But bit closer inspection reveals high probability spots (probability greater than 15%) in some unusual places. There is a very simple explanation for this anomaly. Now, probability of scoring a goal from a particular position is simply:
# 
# <b> Probability of Scoring from a psoition = Total Number of goals from the spot / Total number of shots from the spot </b>
# 
# So now consider a spot very far away from the goal. If very few people attempt a shot from the spot (e.g 5) and one of the shots results in a goals, this returns a probaility of 20% of scoring from the position. Hence, we see high proability goal scoring spots in some unsual areas because the number of shots attempted from those places are very low.
# 
# ## From Which Angles Most Shots Have been Taken and Most Goals Have Been Scored?

# Let us see a histogram of shots frequency from different angles
fig,ax = plt.subplots(figsize = (10,5))
shots_model.hist('Angle', ax = ax)

plt.title('Number of Shots w.r.t Shooting Angle')
plt.xlabel('Shooting Angle')
plt.ylabel('Number of Shots')
plt.savefig('shots_freq.png')
plt.clf()

fig,ax = plt.subplots(figsize = (10,5))
shots_model[shots_model['Goal']== 1].hist('Angle', ax = ax)
plt.title('Number of Goals w.r.t Shooting Angle')
plt.xlabel('Shooting Angle')
plt.ylabel('Number of Goals')
plt.savefig('num_goals_angle.png')
plt.clf()

# ## Goals Scored from Right Foot Comapred to Gaols Scored from Left Foot

# Creating a mask to filter out all the goals
mask = shots_model[shots_model['Goal']== 1]

# Count the number of goals scored by left foot and right foot
left_foot_goals = mask['Left_Foot'].sum()
right_foot_goals = mask['Right_Foot'].sum()

# Plot the results
labels = ['Left Foot', 'Right Foot']
values = [left_foot_goals, right_foot_goals]
plt.bar(labels, values)
plt.title('Number of Goals by Shooting Foot')
plt.xlabel('Foot')
plt.ylabel('Number of Goals')

# Add text labels to the bars
for i, v in enumerate(values):
    plt.text(i, v + 0.25, str(v), color='black', ha='center')
    
plt.savefig('foot.png')
plt.clf()

# ## Goals Scored in 1st Half Comapred to Gaols Scored in Second

# Creating a mask to filter out all the goals
mask = shots_model[shots_model['Goal']== 1]

# Count the number of goals scored by left foot and right foot
FH_goals = mask['1H'].sum()
SH_goals = mask['2H'].sum()

# Plot the results
labels = ['1H Goals', '2H Goals']
values = [FH_goals, SH_goals]

plt.bar(labels, values)
plt.title('Number of Goals by Halves')
plt.xlabel('Match Half')
plt.ylabel('Number of Goals')

# Add text labels to the bars
for i, v in enumerate(values):
    plt.text(i, v + 0.25, str(v), color='black', ha='center')
plt.savefig('half.png')
plt.clf()

# ## From Which Position in Goal Most Number of Goals Went in?

# Creating a mask to filter out all the goals
mask = shots_model[shots_model['Goal']== 1]

# Count the number of goals scored from different posittions
glc_goals = mask['GLC'].sum()
gll_goals = mask['GLL'].sum()
glr_goals = mask['GLR'].sum()
gc_goals  = mask['GC'].sum()
gcr_goals = mask['GCR'].sum()
gcl_goals = mask['GCL'].sum()
ghc_goals = mask['GHC'].sum()
ghr_goals = mask['GHR'].sum()
ghl_goals = mask['GHL'].sum()
# Plot the results
labels = ['GLC', 'GLL','GLR','GC','GCR','GCL','GHC','GHR','GHL']
values = [glc_goals, gll_goals, glr_goals,gc_goals,gcr_goals,gcl_goals,ghc_goals,ghr_goals,ghl_goals]

plt.bar(labels, values)
plt.title('Number of Goals by Positions')
plt.xlabel('Position in Goal')
plt.ylabel('Number of Goals')

# Add text labels to the bars
for i, v in enumerate(values):
    plt.text(i, v + 0.25, str(v), color='black', ha='center')
    
plt.savefig('goals_positions.png')
plt.clf()

# ## Inspecting the Class Imbalance in our Data

# Get the counts of each category
counts = shots_model['Goal'].value_counts()

# Create a bar plot
counts.plot.bar()

# Set the title and axis labels
plt.title('Category Counts')
plt.xlabel('Category')
plt.ylabel('Count')

# Add text labels to the bars
for i, v in enumerate(counts):
    plt.text(i, v + 0.25, str(v), color='black', ha='center')
# Display the plot
plt.savefig('class_imbalance.png')
plt.clf()

# <b> Goal Class "1" </b>:                  3453
# 
# <b> No Goal Class "0" </b>:               30489
# 
# <b> Class Weights </b>: 
# 
#   - Class 1: 0.1017
#   
#   - Class 0: 0.8983

# # <b>Checking for Multi-Collinearity in Data Using VIF Statistic</b>

from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF dataframe
features = shots_model[['X','Y','C','Left_Foot','Right_Foot','GLC', 'GLR', 'GLL', 'GC', 'GCR', 'GCL', 'GHC', 'GHR','GHL','1H', '2H', 'Player_Rank','Distance', 'Angle']]
vif_data = pd.DataFrame()
vif_data["feature"] = features.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(features.values, i)
                          for i in range(len(features.columns))]

pd.options.display.float_format = '{:.4f}'.format

# ### Dealing with Infinity Values
# We are getting the value "inf" (infinity) for 4 columns. Left_Foot, Right_Foot, 1H and 2H. This shows a perfect correlation between two independent variables. In the case of perfect correlation, we get R2 =1, which lead to 1/(1-R2) infinity. To solve this problem we need to drop one of the variables from the dataset which is causing this perfect multicollinearity. This also understandable since we always inlcude n-1 dummy variables for any categorical variable. Right_Foot class 0 represents Left_Foot, similiarly 1H class 0 represents 2H. Therefore, we will drop 1H and Left_Foot from our dataframe.
# 
# ### Dealing with VIF Scores > 10
# X cooridinate and Distance are highly multicollinear, So we will be dropping X from our final dataset.

features = shots_model[['Y','C','Right_Foot','GLC', 'GLR', 'GLL', 'GC', 'GCR', 'GCL', 'GHC', 'GHR','GHL', '2H', 'Player_Rank','Distance', 'Angle']]
vif_data = pd.DataFrame()
vif_data["feature"] = features.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(features.values, i)
                          for i in range(len(features.columns))]

pd.options.display.float_format = '{:.4f}'.format

# Since VIF score for Y coordinate is aprox 10 so we also drop this feature. This will completely remove any trace of multi-collinearity from our data set
features = shots_model[['C','Right_Foot','GLC', 'GLR', 'GLL', 'GC', 'GCR', 'GCL', 'GHC', 'GHR','GHL', '2H', 'Player_Rank','Distance', 'Angle']]
vif_data = pd.DataFrame()
vif_data["feature"] = features.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(features.values, i)
                          for i in range(len(features.columns))]

pd.options.display.float_format = '{:.4f}'.format


# # <b>Train Test Split and Cross Validation </b>

# Creating a dataframe containing all the features
X = shots_model[['C','Right_Foot','GLC', 'GLR', 'GLL', 'GC', 'GCR', 'GCL', 'GHC', 'GHR','GHL', '2H', 'Player_Rank','Distance','Angle']]
# Feature data set

# Scaling the numeric continous features in our data set
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_continuous = X[['C', 'Distance', 'Angle','Player_Rank']] # select only continuous columns
X_scaled = scaler.fit_transform(X_continuous)

# Dropping the original continuous columns in the dataframe
X = X.drop(['C','Player_Rank','Angle','Distance'], axis=1)

# Adding scaled columns to our dataset
X['C'] = X_scaled[:,0]
X['Distance'] = X_scaled[:,1]
X['Angle'] = X_scaled[:,2]
X['Player_Rank'] = X_scaled[:,3]

# Finalized Feature Dataset

# Creating a data set containing dependent variable 'Goal'
y = shots_model['Goal']


# ## Nested Cross-Validation
# 
# We first split the data into training and testing sets using train_test_split() and then use cross-validation on the training set to fine-tune the model's hyperparameters and estimate its performance. In nested cross-validation, the outer loop performs the train-test split and the inner loop performs cross-validation on the training set. The inner loop is used to search for the best hyperparameters for the model, while the outer loop is used to estimate the performance of the model on new data.
# 
# For imbalanced class data, it is generally recommended to use <b>stratified cross-validation</b>, which ensures that each fold has the same proportion of samples from each class as the entire dataset. This can help to ensure that the model is able to learn from examples of the minority class and improve its performance on predicting the minority class. <b> <i> Stratified k-fold cross-validation </i> </b>  is a common method for this, where the data is divided into k folds and each fold has roughly the same proportion of samples from each class.



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score





# # <b> Model # 1:          <u><i>Logistic Regression<i></u> </b>
print("Logistic Regression", file=of)
print("Logistic Regression")
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
count_class_0, count_class_1 = y_train.value_counts()
print ('Shot in Training Data: ', count_class_0, file=of)
print ('Goals in Training Data: ', count_class_1, file=of)
weight_0 = count_class_0 / (count_class_0 + count_class_1)
weight_1 = count_class_1 / (count_class_0 + count_class_1)
print ('Weight of Shots Class in Training: ',weight_0, file=of)
print ('Weight of Goal Class in Training: ', weight_1, file=of)
param_grid_lr = {'C': [0.1, 1, 10, 100],
              'penalty': ['l1', 'l2'],
              'class_weight': ['balanced', {0:weight_0, 1:weight_1}]}
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_model = LogisticRegression()
start_time = time.time()
grid = GridSearchCV(lr_model, param_grid=param_grid_lr, cv=cv_inner, scoring='f1', n_jobs=-1)
scores = cross_val_score(grid, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
best_lr_model = grid.best_estimator_
end_time = time.time()
lr_training_time = end_time - start_time
print("Best parameters: ", grid.best_params_, file=of)
print ("Model Training Time: {:.3f} seconds".format(lr_training_time), file=of)
from tabulate import tabulate
from scipy.stats import norm
coef = best_lr_model.coef_[0]
intercept = best_lr_model.intercept_[0]
# Calculate the standard errors
n = len(y_train)
A = np.hstack((np.ones((n, 1)), X_train))
p = len(coef)
y_pred = best_lr_model.predict(X_train)
residuals = y_train - y_pred
sigma2 = np.sum(residuals**2) / (n - p - 1)
cov = sigma2 * np.linalg.inv(np.dot(A.T, A))
se = np.sqrt(np.diag(cov)[1:])
wald = coef / se
p_values = (1 - norm.cdf(np.abs(wald))) * 2
features = list(X_train.columns)
table = np.column_stack((features, coef, se, wald, p_values))
headers = ['Feature', 'Coef.', 'Std. Err.', 'Wald', 'p-value']
print(tabulate(table, headers=headers), file=of)
cm_train = confusion_matrix(y_train, best_lr_model.predict(X_train))
ax = sns.heatmap(cm_train, annot=True, cmap='Greens', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Train Set')
print (classification_report(y_train, best_lr_model.predict(X_train)), file=of)
y_pred = best_lr_model.predict(X_test)
cm_test = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cm_test, annot=True, cmap='Blues', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
print (classification_report(y_test, y_pred), file=of)



# # <b>Model # 2:          <u><i>XGBClassifier<i></u> </b>
print("XGBClassifier", file=of)
print("XGBClassifier")
import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_model = xgb.XGBClassifier()
count_class_0, count_class_1 = y_train.value_counts()
print ('Shot in Training Data: ', count_class_0, file=of)
print ('Goals in Training Data: ', count_class_1, file=of)
scale_pos_weight = count_class_0 / count_class_1
param_grid_xgb = {'learning_rate': [0.1, 0.01, 0.001],
              'max_depth': [3, 5, 7],
              'n_estimators': [100, 200, 300],
              'scale_pos_weight': [1, scale_pos_weight]}
start_time = time.time()
grid_xg = GridSearchCV(xgb_model, param_grid=param_grid_xgb, cv=cv_inner, scoring='f1', n_jobs=-1)
scores_xg = cross_val_score(grid_xg, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
grid_xg.fit(X_train, y_train)
best_xgb_model = grid_xg.best_estimator_
stop_time = time.time()
xgb_training_time = stop_time - start_time
print("Best parameters: ", grid_xg.best_params_, file=of)
print ("Model Training Time: {:.3f} seconds".format(xgb_training_time), file=of)
cm_train_xg = confusion_matrix(y_train, best_xgb_model.predict(X_train))
ax = sns.heatmap(cm_train_xg, annot=True, cmap='BuPu', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Train Set')
print (classification_report(y_train, best_xgb_model.predict(X_train)), file=of)
y_pred_xgb = best_xgb_model.predict(X_test)
cm_test_xgb = confusion_matrix(y_test, y_pred_xgb)
ax = sns.heatmap(cm_test_xgb, annot=True, cmap='Blues', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
print (classification_report(y_test, y_pred_xgb), file=of)
xgb.plot_importance(best_xgb_model)
plt.savefig('xgb_importance.png')
plt.clf()
xgb.plot_importance(best_xgb_model, importance_type='gain', xlabel='Gain')
plt.savefig('xgb_importance_gain.png')
plt.clf()
xgb.plot_importance(best_xgb_model, importance_type='weight', xlabel='Weight')
plt.savefig('xgb_importance_weight.png')
plt.clf()












# # <b>Model # 3:          <u><i>Random Forests<i></u> </b>
print("Random Forests", file=of)
print("Random Forests")
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
count_class_0, count_class_1 = y_train.value_counts()
weight_0 = count_class_0 / (count_class_0 + count_class_1)
weight_1 = count_class_1 / (count_class_0 + count_class_1)
print ('Weight of Shots Class in Training: ',weight_0, file=of)
print ('Weight of Goal Class in Training: ', weight_1, file=of)
param_grid = {'n_estimators': [100, 200, 300],
              'max_depth': [3, 5, 7,9, 12, 15],
              'min_samples_split': [2, 5, 10],
              'class_weight': ['balanced', {0: weight_0, 1: weight_1}],
			  'random_state': [42]}
rf_model = RandomForestClassifier()
start_time = time.time()
grid_rf = GridSearchCV(rf_model, param_grid=param_grid, cv=cv_inner, scoring='f1', n_jobs=-1)
scores_rf = cross_val_score(grid, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf_model = grid_rf.best_estimator_
end_time = time.time()
rf_training_time = end_time - start_time
print("Best parameters: ", grid_rf.best_params_, file=of)
print ("Model Training Time: {:.3f} seconds".format(rf_training_time), file=of)
cm_train_rf = confusion_matrix(y_train, best_rf_model.predict(X_train))
ax = sns.heatmap(cm_train_rf, annot=True, cmap='YlGnBu', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Train Set')
print (classification_report(y_train, best_rf_model.predict(X_train)), file=of)
y_pred_rf = best_rf_model.predict(X_test)
cm_test_rf = confusion_matrix(y_test, y_pred_rf)
ax = sns.heatmap(cm_test_rf, annot=True, cmap='Blues', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
print (classification_report(y_test, y_pred_rf), file=of)
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X_train.columns[i] for i in indices]
plt.figure()
plt.title("Feature Importance")
sns.barplot(x=importances[indices], y=names)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig('model3.png')
plt.clf()














# # <b>Model # 4:          <u><i>Support Vector Machines<i></u> </b>
print("Support Vector Machines", file=of)
print("Support Vector Machines")
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf', 'sigmoid'],
              'gamma': ['scale', 'auto'],
              'class_weight': ['balanced', {0: weight_0, 1: weight_1}]}
svm_model = SVC()
start_time = time.time()
grid_svm = GridSearchCV(svm_model, param_grid=param_grid, cv=cv_inner, scoring='f1', n_jobs=-1)
scores_svm = cross_val_score(grid_svm, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
grid_svm.fit(X_train, y_train)
best_svm_model = grid_svm.best_estimator_
stop_time = time.time()
svm_training_time = stop_time - start_time
print("Best parameters: ", grid_svm.best_params_, file=of)
print ("Model Training Time: {:.3f} seconds".format(svm_training_time), file=of)
cm_train_svm = confusion_matrix(y_train, best_svm_model.predict(X_train))
ax = sns.heatmap(cm_train_svm, annot=True, cmap='Greens', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Train Set')
print (classification_report(y_train, best_svm_model.predict(X_train)), file=of)
y_pred_svm = best_svm_model.predict(X_test)
cm_test_svm = confusion_matrix(y_test, y_pred_svm)
ax = sns.heatmap(cm_test_svm, annot=True, cmap='Blues', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
print (classification_report(y_test, y_pred_svm), file=of)








# # <b>Model # 5:          <u><i>KNeighborsClassifier<i></u> </b>
print("KNeighborsClassifier", file=of)
print("KNeighborsClassifier")
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
knn_model = KNeighborsClassifier()
start_time = time.time()
grid_knn = GridSearchCV(knn_model, param_grid=param_grid, cv=cv_inner, scoring='f1', n_jobs=-1)
scores_knn = cross_val_score(grid_knn, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
grid_knn.fit(X_train, y_train)
best_knn_model = grid_knn.best_estimator_
stop_time = time.time()
knn_training_time = stop_time - start_time
print("Best parameters: ", grid_knn.best_params_, file=of)
print ("Model Training Time: {:.3f} seconds".format(knn_training_time), file=of)
cm_train_knn = confusion_matrix(y_train, best_knn_model.predict(X_train))
ax = sns.heatmap(cm_train_knn, annot=True, cmap='Greens', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Train Set')
print (classification_report(y_train, best_knn_model.predict(X_train)), file=of)
y_pred_knn = best_knn_model.predict(X_test)
cm_test_knn = confusion_matrix(y_test, y_pred_knn)
ax = sns.heatmap(cm_test_knn, annot=True, cmap='Blues', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
print (classification_report(y_test, y_pred_knn), file=of)







# # <b>Model # 6:          <u><i>LightGBM<i></u> </b>
print("LightGBM", file=of)
print("LightGBM")
import lightgbm as lgb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'n_estimators': [10, 100, 1000], 'num_leaves': [3, 31, 101], 'max_depth': [-1, 5, 10]}
lgb_model = lgb.LGBMClassifier()
start_time = time.time()
grid_lgb = GridSearchCV(lgb_model, param_grid=param_grid, cv=cv_inner, scoring='f1', n_jobs=-1)
scores_lgb = cross_val_score(grid_lgb, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
grid_lgb.fit(X_train, y_train)
best_lgb_model = grid_lgb.best_estimator_
stop_time = time.time()
lgb_training_time = stop_time - start_time
print("Best parameters: ", grid_lgb.best_params_, file=of)
print ("Model Training Time: {:.3f} seconds".format(lgb_training_time), file=of)
cm_train_lgb = confusion_matrix(y_train, best_lgb_model.predict(X_train))
ax = sns.heatmap(cm_train_lgb, annot=True, cmap='Greens', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Train Set')
print (classification_report(y_train, best_lgb_model.predict(X_train)), file=of)
y_pred_lgb = best_lgb_model.predict(X_test)
cm_test_lgb = confusion_matrix(y_test, y_pred_lgb)
ax = sns.heatmap(cm_test_lgb, annot=True, cmap='Blues', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
print (classification_report(y_test, y_pred_lgb), file=of)






# # <b>Model # 7:          <u><i>DecisionTreeClassifier<i></u> </b>
print("DecisionTreeClassifier", file=of)
print("DecisionTreeClassifier")
from sklearn import tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'max_depth': [1, 2, 3, 4, 5]}
dtc_model = tree.DecisionTreeClassifier()
start_time = time.time()
grid_dtc = GridSearchCV(dtc_model, param_grid=param_grid, cv=cv_inner, scoring='f1', n_jobs=-1)
scores_dtc = cross_val_score(grid_dtc, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
grid_dtc.fit(X_train, y_train)
best_dtc_model = grid_dtc.best_estimator_
stop_time = time.time()
dtc_training_time = stop_time - start_time
print("Best parameters: ", grid_dtc.best_params_, file=of)
print ("Model Training Time: {:.3f} seconds".format(dtc_training_time), file=of)
cm_train_dtc = confusion_matrix(y_train, best_dtc_model.predict(X_train))
ax = sns.heatmap(cm_train_dtc, annot=True, cmap='Greens', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Train Set')
print (classification_report(y_train, best_dtc_model.predict(X_train)), file=of)
y_pred_dtc = best_dtc_model.predict(X_test)
cm_test_dtc = confusion_matrix(y_test, y_pred_dtc)
ax = sns.heatmap(cm_test_dtc, annot=True, cmap='Blues', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
print (classification_report(y_test, y_pred_dtc), file=of)







# # <b>Model # 8:          <u><i>CatBoostClassifier<i></u> </b>
print("CatBoostClassifier", file=of)
print("CatBoostClassifier")
from catboost import CatBoostClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'iterations': [1, 5, 10, 50, 100], 'learning_rate': [0.01, 0.1, 1], 'depth': [-1, 5, 10, 100]}
cbc_model = CatBoostClassifier()
start_time = time.time()
grid_cbc = GridSearchCV(cbc_model, param_grid=param_grid, cv=cv_inner, scoring='f1', n_jobs=-1)
scores_cbc = cross_val_score(grid_cbc, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
grid_cbc.fit(X_train, y_train)
best_cbc_model = grid_cbc.best_estimator_
stop_time = time.time()
cbc_training_time = stop_time - start_time
print("Best parameters: ", grid_cbc.best_params_, file=of)
print ("Model Training Time: {:.3f} seconds".format(cbc_training_time), file=of)
cm_train_cbc = confusion_matrix(y_train, best_cbc_model.predict(X_train))
ax = sns.heatmap(cm_train_cbc, annot=True, cmap='Greens', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Train Set')
print (classification_report(y_train, best_cbc_model.predict(X_train)), file=of)
y_pred_cbc = best_cbc_model.predict(X_test)
cm_test_cbc = confusion_matrix(y_test, y_pred_cbc)
ax = sns.heatmap(cm_test_cbc, annot=True, cmap='Blues', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
print (classification_report(y_test, y_pred_cbc), file=of)








# # <b>Model # 9:          <u><i>ANN<i></u> </b>
print("ANN", file=of)
print("ANN")
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier

def build_clf(hidden_layer_dim, meta):
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]
    model = Sequential()
    model.add(Dense(n_features_in_, input_shape=X_shape_[1:]))
    model.add(Activation("relu"))
    model.add(Dense(hidden_layer_dim))
    model.add(Activation("relu"))
    model.add(Dense(n_classes_))
    model.add(Activation("softmax"))
    return model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ann_model = KerasClassifier(
    build_clf,
    loss="sparse_categorical_crossentropy",
    hidden_layer_dim=100,
)
param_grid_ann = {
    "hidden_layer_dim": [50, 100, 200],
    "loss": ["sparse_categorical_crossentropy"],
    "optimizer": ["adam", "sgd"],
    "optimizer__learning_rate": [0.0001, 0.001, 0.1]
}
start_time = time.time()
grid_ann = GridSearchCV(ann_model, param_grid=param_grid_ann, cv=cv_inner, scoring='f1', n_jobs=-1)
scores_ann = cross_val_score(grid_ann, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
grid_ann.fit(X_train, y_train)
best_ann_model = grid_ann.best_estimator_
stop_time = time.time()
ann_training_time = stop_time - start_time
print("Best parameters: ", grid_ann.best_params_, file=of)
print ("Model Training Time: {:.3f} seconds".format(ann_training_time), file=of)
cm_train_ann = confusion_matrix(y_train, best_ann_model.predict(X_train))
ax = sns.heatmap(cm_train_ann, annot=True, cmap='Greens', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Train Set')
print (classification_report(y_train, best_ann_model.predict(X_train)), file=of)
y_pred_ann = best_ann_model.predict(X_test)
cm_test_ann = confusion_matrix(y_test, y_pred_ann)
ax = sns.heatmap(cm_test_ann, annot=True, cmap='Blues', fmt='g', linewidth=1.5)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
print (classification_report(y_test, y_pred_ann), file=of)









# # <b>Summary and Comparison of Classification Results</b>
# Precision scores for the nine models on training data
prec_lr_train = precision_score(y_train, best_lr_model.predict(X_train))
prec_xgb_train = precision_score(y_train, best_xgb_model.predict(X_train))
prec_rf_train = precision_score(y_train, best_rf_model.predict(X_train))
prec_svm_train = precision_score(y_train, best_svm_model.predict(X_train))
prec_knn_train = precision_score(y_train, best_knn_model.predict(X_train))
prec_lgb_train = precision_score(y_train, best_lgb_model.predict(X_train))
prec_dtc_train = precision_score(y_train, best_dtc_model.predict(X_train))
prec_cbc_train = precision_score(y_train, best_cbc_model.predict(X_train))
prec_ann_train = precision_score(y_train, best_ann_model.predict(X_train))
prec_train = [prec_lr_train, prec_xgb_train, prec_rf_train, prec_svm_train, prec_knn_train, prec_lgb_train, prec_dtc_train, prec_cbc_train, prec_ann_train]

# Precision scores for the nine models on testing data
prec_lr_test = precision_score(y_test, y_pred)
prec_xgb_test = precision_score(y_test, y_pred_xgb)
prec_rf_test = precision_score(y_test, y_pred_rf)
prec_svm_test = precision_score(y_test, y_pred_svm)
prec_knn_test = precision_score(y_test, y_pred_knn)
prec_lgb_test = precision_score(y_test, y_pred_lgb)
prec_dtc_test = precision_score(y_test, y_pred_dtc)
prec_cbc_test = precision_score(y_test, y_pred_cbc)
prec_ann_test = precision_score(y_test, y_pred_ann)
prec_test = [prec_lr_test, prec_xgb_test, prec_rf_test, prec_svm_test, prec_knn_test, prec_lgb_test, prec_dtc_test, prec_cbc_test, prec_ann_test]

# Recall scores for the nine models on training data
rec_lr_train = recall_score(y_train, best_lr_model.predict(X_train))
rec_xgb_train = recall_score(y_train, best_xgb_model.predict(X_train))
rec_rf_train = recall_score(y_train, best_rf_model.predict(X_train))
rec_svm_train = recall_score(y_train, best_svm_model.predict(X_train))
rec_knn_train = recall_score(y_train, best_knn_model.predict(X_train))
rec_lgb_train = recall_score(y_train, best_lgb_model.predict(X_train))
rec_dtc_train = recall_score(y_train, best_dtc_model.predict(X_train))
rec_cbc_train = recall_score(y_train, best_cbc_model.predict(X_train))
rec_ann_train = recall_score(y_train, best_ann_model.predict(X_train))
rec_train = [rec_lr_train, rec_xgb_train, rec_rf_train, rec_svm_train, rec_knn_train, rec_lgb_train, rec_dtc_train, rec_cbc_train, rec_ann_train]

# Calculting Recall for the nine models on test data
rec_lr_test = recall_score(y_test, y_pred)
rec_xgb_test = recall_score(y_test, y_pred_xgb)
rec_rf_test = recall_score(y_test, y_pred_rf)
rec_svm_test = recall_score(y_test, y_pred_svm)
rec_knn_test = recall_score(y_test, y_pred_knn)
rec_lgb_test = recall_score(y_test, y_pred_lgb)
rec_dtc_test = recall_score(y_test, y_pred_dtc)
rec_cbc_test = recall_score(y_test, y_pred_cbc)
rec_ann_test = recall_score(y_test, y_pred_ann)
rec_test = [rec_lr_test, rec_xgb_test, rec_rf_test, rec_svm_test, rec_knn_test, rec_lgb_test, rec_dtc_test, rec_cbc_test, rec_ann_test]

# Accuracy scores for the nine models on training data
acc_lr_train = accuracy_score(y_train, best_lr_model.predict(X_train))
acc_xgb_train = accuracy_score(y_train, best_xgb_model.predict(X_train))
acc_rf_train = accuracy_score(y_train, best_rf_model.predict(X_train))
acc_svm_train = accuracy_score(y_train, best_svm_model.predict(X_train))
acc_knn_train = accuracy_score(y_train, best_knn_model.predict(X_train))
acc_lgb_train = accuracy_score(y_train, best_lgb_model.predict(X_train))
acc_dtc_train = accuracy_score(y_train, best_dtc_model.predict(X_train))
acc_cbc_train = accuracy_score(y_train, best_cbc_model.predict(X_train))
acc_ann_train = accuracy_score(y_train, best_ann_model.predict(X_train))
acc_train = [acc_lr_train, acc_xgb_train, acc_rf_train, acc_svm_train, acc_knn_train, acc_lgb_train, acc_dtc_train, acc_cbc_train, acc_ann_train]

# Calculating Accuracy for the nine models on test data
acc_lr_test = accuracy_score(y_test, y_pred)
acc_xgb_test = accuracy_score(y_test, y_pred_xgb)
acc_rf_test = accuracy_score(y_test, y_pred_rf)
acc_svm_test = accuracy_score(y_test, y_pred_svm)
acc_knn_test = accuracy_score(y_test, y_pred_knn)
acc_lgb_test = accuracy_score(y_test, y_pred_lgb)
acc_dtc_test = accuracy_score(y_test, y_pred_dtc)
acc_cbc_test = accuracy_score(y_test, y_pred_cbc)
acc_ann_test = accuracy_score(y_test, y_pred_ann)
acc_test = [acc_lr_test, acc_xgb_test, acc_rf_test, acc_svm_test, acc_knn_test, acc_lgb_test, acc_dtc_test, acc_cbc_test, acc_ann_test]

train_time =[lr_training_time/60, xgb_training_time/60, rf_training_time/60, svm_training_time/60, knn_training_time/60, lgb_training_time/60, dtc_training_time/60, cbc_training_time/60, ann_training_time/60]
summary_df = pd.DataFrame({'Model Name':['Logistic Regression','XG Boost','Random Forests','SVM','K-nearest Neighbors','Decision Trees Classifier','LightGBM','CatBoost', 'ANN'],'Training Accuracy': acc_train, 'Training Precision': prec_train,'Training Recall':rec_train,'Testing Accuracy': acc_test,'Testing Precision': prec_test,'Testing Recall':rec_test,'Training Time (mins)': train_time})
summary_df.set_index('Model Name', inplace=True)
summary_df.style.format(precision =3).highlight_max(color='cyan').set_properties(**{'font-weight': 'bold', 'border': '2.0px solid grey','color': 'purple'}).highlight_min(color='yellow')

print(summary_df.to_string(), file=of)

of.close()
print('\nFinished!')