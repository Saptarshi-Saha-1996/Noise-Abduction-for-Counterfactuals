import os
directory = 'datasets/counterfactuals'
 
# iterate over files in
# that directory
dictionary_full={}
dictionary_partial={}

index_full=0
index_partial=0

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        g=f.split('/')[-1].split('.')[0].split('_')
        key={'path':f,'flowtype':g[-6],'floworder':g[-4],'intervention':  g[-3]+'('+g[-2]+'='+g[-1]+')'}
        if g[1]=='path':
            model='partial'
            key['model']=model
            dictionary_partial[index_partial]=key
            index_partial=index_partial+1
        else:
            model='full'
            key['model']=model
            dictionary_full[index_full]=key
            index_full=index_full+1

item=[]
for i in range(12):
    for j in range(12):
        if dictionary_full[i]['flowtype']==dictionary_partial[j]['flowtype']:
            if dictionary_full[i]['floworder']==dictionary_partial[j]['floworder'] :
                if dictionary_full[i]['intervention']==dictionary_partial[j]['intervention']:
                    item.append((i,j))

item_full=[]                
for i in range(12):
    for j in range(12):
        if dictionary_full[i]['flowtype']==dictionary_full[j]['flowtype']:
            if dictionary_full[i]['floworder']==dictionary_full[j]['floworder'] :
                if dictionary_full[i]['intervention']!=dictionary_full[j]['intervention']:
                    item_full.append((i,j))



item_full_updated=[]

for(i,j) in item_full:
    if (j,i) in item_full and i<j:
        item_full_updated.append((i,j))
item_full=item_full_updated
        
flow_combinations=[]
for (i,j) in item:
    for (k,l) in item_full:
        if i==k :
            m=[t for (s,t) in item if s == l][0]
            flow_combinations.append([(i,j),(l,m)])


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_file(path):
    credit=pd.read_csv(path)
    credit=credit=pd.DataFrame(data=credit, columns=["Index","Sex", "Age","Credit amount","Duration","Default"])
    return credit



    
def plot_function(flow_combination):
    [(k,l),(m,n)]=flow_combination
    #print(k,l,m,n)
    k=dictionary_full[k]
    l=dictionary_partial[l]
    m=dictionary_full[m]
    n=dictionary_partial[n]
    
    if k['flowtype']==l['flowtype']==m['flowtype']==n['flowtype']:
        flowtype=k['flowtype']
    #print(flowtype)
    
    if k['floworder']==l['floworder']==m['floworder']==n['floworder']:
        floworder=k['floworder']
    #print(floworder)
    
    
    x_k=[k,read_file(k['path']).assign(model=k['model'])]
    x_l=[l,read_file(l['path']).assign(model=l['model'])]
    x_m=[m,read_file(m['path']).assign(model=m['model'])]
    x_n=[n,read_file(n['path']).assign(model=n['model'])]
    default=read_file('data/german_credit_data.csv').assign(model='observed')
    
    do_sex_male=[]
    do_sex_female=[]
    for i in [x_k,x_l,x_m,x_n]:
        if i[0]['intervention']=='do(sex=Female)':
               do_sex_female.append(i[1])
        if i[0]['intervention']=='do(sex=Male)':
            do_sex_male.append(i[1])
            
                
    default_male=default.loc[default['Sex']=='male']

    do_sex_male[1]= do_sex_male[1].loc[default['Sex']=='female']
    do_sex_male[0]= do_sex_male[0].loc[default['Sex']=='female']
    default_female=default.loc[default['Sex']=='female']
    
    
    do_sex_female[1]= do_sex_female[1].loc[default['Sex']=='male']
    do_sex_female[0]= do_sex_female[0].loc[default['Sex']=='male']
    default_male=default.loc[default['Sex']=='male']
    
    did_sex_female = pd.concat([ do_sex_female[1], do_sex_female[0],default_male])    
    do_sex_female = pd.melt(did_sex_female, id_vars=["Index",'Sex', "Age","Credit amount","Duration",'model'],var_name=['Default'])
    #print(do_sex_female)
    
    did_sex_male = pd.concat([ do_sex_male[1], do_sex_male[0],default_female])    
    do_sex_male = pd.melt(did_sex_male, id_vars=["Index",'Sex', "Age","Credit amount","Duration",'model'],var_name=['Default'])
    #print(do_sex_male)
    
    sns.set(style="ticks")
    
    fig, ax = plt.subplots(2,2, figsize=(18,12))
    sns.kdeplot(data=do_sex_female,x='Credit amount',hue='model',palette={'red','black','gray'},ax=ax[0][0],fill=True,alpha=0.4,legend=True).set_title("do(Sex=Female)",fontweight='bold')
    sns.move_legend(ax[0][0],loc='lower right')
    sns.kdeplot(data=do_sex_female,x='Duration',hue='model',ax=ax[1][0],palette={'red','black','gray'},fill=True,alpha=0.5,legend=True).set_title("do(Sex=Female)",fontweight='bold')
    sns.move_legend(ax[1][0],loc='lower right')
    ax_inside=ax[0][0].inset_axes([0.5,0.5,0.5,0.5])
    ax_inside_1=ax[1][0].inset_axes([0.5,0.5,0.5,0.5])
    sns.boxplot(data=do_sex_female,x='model',y='Credit amount',hue='Sex' ,palette={'red','gray'},linewidth=2.5,ax=ax_inside)
    ax_inside.legend(loc='upper left')
    sns.boxplot(data=do_sex_female,x='model',y='Duration',hue='Sex',palette={'red','gray'},linewidth=2.5,ax=ax_inside_1)
    ax_inside_1.legend(loc='upper left')
    
    sns.kdeplot(data=do_sex_male,x='Credit amount',hue='model',ax=ax[0][1],fill=True,palette={'red','gray','black'},alpha=0.4,legend=True).set_title("do(Sex=Male)",fontweight='bold')
    sns.move_legend(ax[0][1],loc='lower right')
    sns.kdeplot(data=do_sex_male,x='Duration',hue='model',ax=ax[1][1],fill=True,palette={'red','black','gray'},alpha=0.5,legend=True).set_title("do(Sex=Male)",fontweight='bold')
    sns.move_legend(ax[1][1],loc='lower right')
    ax_inside_2=ax[0][1].inset_axes([0.5,0.5,0.5,0.5])
    ax_inside_3=ax[1][1].inset_axes([0.5,0.5,0.5,0.5])
    sns.boxplot(data=do_sex_male,x='model',y='Credit amount',hue='Sex',palette={'red','gray'},linewidth=2,ax=ax_inside_2)
    ax_inside_2.legend(loc='upper left')
    sns.boxplot(data=do_sex_male,x='model',y='Duration',hue='Sex',palette={'red','gray'},linewidth=2,ax=ax_inside_3)
    ax_inside_3.legend(loc='upper left')
    fig.suptitle('flowtype: '+flowtype+', floworder: '+floworder,y=0.05,fontweight='bold' )
    
    return fig, flowtype,floworder
    
        
#x=plot_function(flow_combinations[0])   

def save_png(x):
    directory='assets/plots'
    if not os.path.exists(directory):
        os.mkdir(directory)
    x[0].savefig(os.path.join(directory,x[1]+"_"+x[2]+'.pdf'), format='pdf',pad_inches=0.1,bbox_inches='tight',dpi=1200)
    
    
if __name__ == '__main__':
    for pair in flow_combinations:
        save_png(plot_function(pair))