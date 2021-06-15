#!/usr/bin/env python
# coding: utf-8

# In[390]:


import numpy as np
import pandas as pd
from datetime import datetime
import networkx as nx
from scipy.special import softmax
from tqdm import tqdm
from random import shuffle
import copy
from collections import defaultdict
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('nyc_citi.csv')
data = data.loc[[type(i)==str for i in data['start station name']]]
data = data.loc[[type(i)==str for i in data['end station name']]]
data.head()


# In[3]:


data.starttime=data.starttime.apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
data.stoptime=data.stoptime.apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
print(data.shape)


# In[4]:


def euc_dis(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**.5


# In[74]:


def build_station_graph(data,starttime,stoptime):
    graphx=nx.Graph()
    filtered_data = data[(data['starttime']>=starttime) & (data['stoptime']<=stoptime)]
    locations = list(set(filtered_data['start station name']).union(filtered_data['end station name']))
    vertices = pd.Series([{}]*len(locations),index=locations)
    for loc in locations:
        vertices[loc] = {'change':0,'x':np.random.random(),'y':np.random.random(),'type':''}
        
    for i in filtered_data['start station name']:
        vertices[i]['change'] -= 1
    for i in filtered_data['end station name']:
        vertices[i]['change'] += 1
    for loc in locations:
        if vertices[loc]['change'] > 0:
            vertices[loc]['type'] = 'overflow'
        if vertices[loc]['change'] < 0:
            vertices[loc]['type'] = 'underflow'
    vertices = vertices.loc[[x['type'] != '' for x in vertices]]
    for vertex in vertices.index:
        graphx.add_node(vertex,bipartite=int(vertices[vertex]['type']=='overflow'))
    #print(vertices)
    nx.set_node_attributes(graphx, {i: vertices[i] for i in vertices.index}) 
    for i in range(len(vertices.index)-1):
        for j in range(len(vertices.index[i:])):
            if vertices[vertices.index[i]]['type'] != vertices[vertices.index[j]]['type']:
                x1=vertices[vertices.index[i]]['x']
                y1=vertices[vertices.index[i]]['y']
                x2=vertices[vertices.index[j]]['x']
                y2=vertices[vertices.index[j]]['y']
                graphx.add_weighted_edges_from([(vertices.index[i],vertices.index[j],euc_dis(x1,y1,x2,y2))])
    return graphx

def cloned_station_vertices(graphx):
    nodes = list(graphx.nodes())
    for node in nodes:
        if graphx.nodes[node]['change']>1:
            for i in range(graphx.nodes[node]['change']-1):
                i = i+1
                graphx.add_nodes_from([(str(node)+str(i),graphx.nodes[node])],bipartite=int(graphx.nodes[node]['type']=='underflow'))
                #print(graphx[str(node)+str(i)])
                new_node_name = str(node)+str(i)
                new_node_data = graphx.nodes[new_node_name]
                for node_ in graphx.nodes:
                    if np.sign(new_node_data['change']) != np.sign(graphx.nodes[node_]['change']):
                        x1=new_node_data['x']
                        y1=new_node_data['y']
                        x2=graphx.nodes[node_]['x']
                        y2=graphx.nodes[node_]['y']
                        graphx.add_weighted_edges_from([(node_,new_node_name,euc_dis(x1,y1,x2,y2))])
    return graphx
                


                    
            


# In[181]:


graph = build_station_graph(data,data.loc[0,'starttime'],data.loc[10,'stoptime'])
#print(graph['E 6 St & Avenue D'])


cgraph = cloned_station_vertices(graph)
workers = {i:{
                'type':'worker',
               'xs':np.random.random(),
               'ys':np.random.random(),
               'xe':np.random.random(),
               'ye':np.random.random()
              } for i in range(np.min([len(underflow), len(overflow)]))}
cwgraph = cgraph.copy()
print(cwgraph.nodes)
cwgraph.add_nodes_from(range(len(workers)))
nx.set_node_attributes(cwgraph, workers) 
cwgraph.add_edges_from([(w,s) for w in workers for s in cgraph.nodes ])
print(cwgraph.nodes)
print(cwgraph.nodes[1])


# In[435]:



    

class UGA:
    def __init__(self,meta_graph):
        self.meta_graph = meta_graph
        self.overflow = [node for node in meta_graph.nodes() if meta_graph.nodes[node]['type'] == 'overflow']
        self.worker = [node for node in meta_graph.nodes() if meta_graph.nodes[node]['type'] == 'worker']
        self.underflow = [node for node in meta_graph.nodes() if meta_graph.nodes[node]['type'] == 'underflow']
        
    def add_triplet(self, vertex, graph):
        neighbors = list(self.meta_graph.neighbors(vertex))
        shuffle(neighbors)
        overflow = None
        underflow = None
        for overflow_ in neighbors:
            if (not overflow_ in list(graph.nodes())) and self.meta_graph.nodes[overflow_]['type']=='overflow':
                
                overflow = overflow_
        for underflow_ in neighbors:
            if (not underflow_ in list(graph.nodes())) and self.meta_graph.nodes[underflow_]['type']=='underflow':
                underflow = underflow_
        if overflow != None and underflow != None:    
            graph.add_node(overflow)
            graph.add_node(underflow)
            graph.add_node(vertex)
            graph.add_edge(vertex,underflow)
            graph.add_edge(vertex,overflow)
            
    
            
    #each species is a complete of worker station triples
    def build_matching(self):
        graph = nx.Graph()
        
        shuffle(self.worker)
        for w in self.worker:
            self.add_triplet(w,graph)
        graph.add_nodes_from([i for i in self.meta_graph.nodes if not i in list(graph.nodes)])
        return graph
    
    def find_overflow(self,vertex):
        while True:
            node_ind = np.random.choice(range(len(self.overflow)))
            if self.overflow[node_ind] != vertex:
                return self.overflow[node_ind]
            
    def find_underflow(self,vertex):
        while True:
            node_ind = np.random.choice(range(len(self.underflow)))
            if self.underflow[node_ind] != vertex:
                return self.underflow[node_ind]
            
    
    def swap(self, worker, to_replace, graph):
        if self.meta_graph.nodes[to_replace]['type'] == 'overflow':
            other = self.find_overflow(to_replace)
        else:
            other = self.find_underflow(to_replace)
        if len(list(graph.neighbors(other)))>0:
            otherw = list(graph.neighbors(other))[0]
            
            graph.add_edge(otherw,to_replace)
            graph.remove_edge(otherw,other)
        graph.add_edge(worker,other)
        graph.remove_edge(worker,to_replace)
            
        
        
    def euc_fitness(self,spec):
        return sum([self.euc_fitness_ind(gene,spec) for gene in spec.nodes if self.meta_graph.nodes[gene]['type'] == 'worker'])

    def euc_fitness_ind(self,gene,spec):
        w = 0


        neighbors = list(spec.neighbors(gene))
        if len(neighbors) != 0:
            for i in neighbors:
                if self.meta_graph.nodes[i]['type'] == 'underflow':
                    w += euc_dis(self.meta_graph.nodes[i]['x'], self.meta_graph.nodes[i]['y'],
                                 self.meta_graph.nodes[gene]['xs'],self.meta_graph.nodes[gene]['ys'])
                else:
                    w += euc_dis(self.meta_graph.nodes[i]['x'],self.meta_graph.nodes[i]['y'],
                                 self.meta_graph.nodes[gene]['xe'],self.meta_graph.nodes[gene]['ye'])
            w += euc_dis(self.meta_graph.nodes[neighbors[0]]['x'],self.meta_graph.nodes[neighbors[0]]['y'],
                         self.meta_graph.nodes[neighbors[1]]['x'],self.meta_graph.nodes[neighbors[1]]['y'])

        return -w

    def compare_specs(self,s1,s2,fitness):
        return self.euc_fitness(s1) > self.euc_fitness(s2)

    
            
    def run(self, sswap_rate, oswap_rate, gens, pop_size):
        #make a population
        pop = [self.build_matching() for i in range(pop_size)]
        
        bests = []
        for gen in tqdm(range(gens)):
            #get scores of species
            scores = np.array([self.euc_fitness(s) for s in pop])
            bests.append(max(scores))
            
            scores = softmax(scores)  
            print(scores[:20])
            selected = []
            
            #while selecting pick a species
            while len(selected) < len(pop):
                spec_ind = np.random.choice(list(range(len(pop))), p=scores)
                spec = pop[spec_ind]
                #for each gene (a triple)
                for j,node in enumerate(spec.nodes):
                    #swap with another pop
                    edges = set(spec.edges)
                    if self.meta_graph.nodes[node]['type']!= 'worker' and len(list(spec.neighbors(node)))>0 and np.random.random() < sswap_rate:
                        self.swap(list(spec.neighbors(node))[0], node, spec)
                        print(edges.intersection(set(self.meta_graph.edges).intersection(set(spec.edges))))
                selected.append(spec)
                                    
            pop = selected
        return pop[np.argmax(scores)],np.max(scores), bests
                
                
    
                            


# In[436]:


#[cgraph.nodes()[node] for node in cgraph.nodes()]


# In[ ]:


uga = UGA(cwgraph)

best, score, log = uga.run(0.5,0.001,400,100)


# In[431]:


plt.plot(log)
plt.show()


# In[469]:


len(overflow)


# In[85]:



edge_list = defaultdict(lambda:{})
for edge in cgraph.edges:
    if graph.edges[edge]['weight'] > 0 and cgraph.nodes[edge[0]]['type'] != 'worker' and cgraph.nodes[edge[0]]['type'] != 'worker':
        edge_list[edge[0]][edge[1]] = graph.edges[edge]['weight']
        edge_list[edge[1]][edge[0]] = graph.edges[edge]['weight']
#print(edge_list)

overflow = {node:{} for node in graph.nodes() if graph.nodes[node]['type']=='overflow'}
underflow = {node:{} for node in graph.nodes() if graph.nodes[node]['type']=='underflow'}
worker = {node:{} for node in cwgraph.nodes() if cwgraph.nodes[node]['type']=='worker'}


# In[84]:



matching = bipartite.matching.minimum_weight_full_matching(cgraph)
spwgraph =nx.Graph()
for key in matching.keys():
    if cgraph.nodes[key]['type'] =='overflow':
        for w, worker_ in enumerate(worker):
            spwgraph.add_edge((key+'|'+matching[key]),w,
                              weight=cgraph.edges[key, matching[key]]['weight']+
                             euc_dis(cwgraph.nodes[worker_]['xs'],cwgraph.nodes[worker_]['ys'],cgraph.nodes[matching[key]]['x'],cgraph.nodes[matching[key]]['y']) +
                             euc_dis(cwgraph.nodes[worker_]['xe'],cwgraph.nodes[worker_]['ye'],cgraph.nodes[matching[key]]['x'],cgraph.nodes[matching[key]]['y'])
                                         )
matching2 = bipartite.matching.minimum_weight_full_matching(spwgraph)
matching2


# In[29]:


print(graph.nodes['Maodison St & Montgomery St'])


# In[680]:


cgraph.edges['Rivington St & Chrystie St', 'Rivington St & Chrystie St']


# In[ ]:




