from collections import defaultdict
from matplotlib import colors
import numpy as np

def plot_embeddings(ax, X, wrd2idx, color_dict=None, default_color="black",show_labels=False,
                    jitter=1,point_size=100,label_size=10,alpha=0.8):
    items = wrd2idx.keys()
    idxs  = wrd2idx.values()
    #reorder matrix to ensure consistency between items and assigned colors
    X = X[idxs]    
    vis_x = X[:, 0]
    vis_y = X[:, 1]
    #add some jitter
    vis_x += np.random.rand(len(vis_x)) * jitter
    vis_y += np.random.rand(len(vis_y)) * jitter
    
    if color_dict is None:
    	color_dict = {"":default_color}
    #add a default color (using a default dictionary)
    color_dict = defaultdict(lambda:default_color,color_dict)
    print "fo shizle"
    k = [x + 0.5 for x in range(len(color_dict))]
    k+=[max(k)+1]        
    cmap, norm = colors.from_levels_and_colors(k,color_dict.values()) 
    ax.set_axis_bgcolor('whitesmoke')
    ax.scatter(vis_x, vis_y, c=idxs,cmap=cmap,s=point_size,alpha=alpha)
    if show_labels:
        for i, x, y in zip(items,list(vis_x),list(vis_y)):
            x = float(x)+label_size*1.0/3
            y = float(y)+label_size*1.0/3
            ax.annotate(i,(x,y),color=default_color,size=label_size) 
    else:
        label_size=0
    #adjustments
    padding=25
    #add extra space for the text labels
    padding+=label_size
    x_max = np.max(vis_x)+padding
    x_min = np.min(vis_x)-padding
    y_max = np.max(vis_y)+padding
    y_min = np.min(vis_y)-padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([], [])
    ax.set_yticks([], [])    
    
    return ax

def plot_ranking(ax,rank,items,color_dict,default_color="red",legend=False):    

    if color_dict is None:
        color_dict = {"":default_color}
    #add a default color (using a default dictionary)
    color_dict = defaultdict(lambda:default_color,color_dict)
    k = [x - 0.4 for x in range(len(color_dict))]
    k+=[max(k) - 0.01]        
    cmap, norm = colors.from_levels_and_colors(k,color_dict.values())     
    
    ax.pcolor(rank[::-1],cmap = cmap, norm = norm)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlim(0,rank.shape[1])