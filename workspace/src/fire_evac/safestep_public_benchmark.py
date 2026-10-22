"""Common legal-action evaluator for the public SafeStep hazard-rate table.

The time-expanded minimum-dose planner has future information. It is a finite
horizon reference for the specified FED objective, not an online competitor or
an optimum of the original DQN reward (which also depends on visit history).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math

import numpy as np

DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class Graph:
    cells: list[tuple[int, int]]
    exits: tuple[tuple[int, int], ...]

    def __post_init__(self):
        self.cells = [tuple(c) for c in self.cells]
        self.exits = tuple(tuple(c) for c in self.exits)
        if len(set(self.cells)) != len(self.cells) or not set(self.exits) <= set(self.cells):
            raise ValueError('Unique cells containing all exits required')
        self.index = {c:i for i,c in enumerate(self.cells)}
        self.exit_ids = {self.index[c] for c in self.exits}
        self.edges = [[] for _ in self.cells]
        self.incoming = [[] for _ in self.cells]
        for i,(x,y) in enumerate(self.cells):
            if i in self.exit_ids:
                continue  # terminal on first arrival
            for action,(dx,dy) in enumerate(DELTAS):
                j = self.index.get((x+dx,y+dy))
                if j is not None:
                    self.edges[i].append((action,j))
                    self.incoming[j].append((action,i))

    def destination(self, cell_id, action):
        for a,j in self.edges[cell_id]:
            if a == action:
                return j
        raise ValueError('Illegal action; evaluator never silently waits')


@dataclass
class RateField:
    times: np.ndarray
    rates: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times,dtype=float)
        self.rates = np.asarray(self.rates,dtype=float)
        if (self.times.ndim != 1 or len(self.times)<2 or self.rates.ndim != 2
                or self.rates.shape[0] != len(self.times)
                or not np.isfinite(self.times).all() or not np.isfinite(self.rates).all()
                or np.any(np.diff(self.times)<=0) or np.any(self.rates<0)):
            raise ValueError('Finite, nonnegative rates and increasing native times required')
        widths = np.diff(self.times)[:,None]
        self.slopes = np.diff(self.rates,axis=0)/widths
        areas = widths*(self.rates[:-1]+self.rates[1:])/2
        self.prefix = np.concatenate((np.zeros((1,self.rates.shape[1])),np.cumsum(areas,axis=0)))

    def _position(self,t):
        if not math.isfinite(t) or t<self.times[0] or t>self.times[-1]:
            raise ValueError('No extrapolation beyond public rate coverage')
        return min(int(np.searchsorted(self.times,t,side='right'))-1,len(self.times)-2)

    def sample(self,t):
        i=self._position(t)
        return self.rates[i]+self.slopes[i]*(t-self.times[i])

    def primitive(self,t):
        i=self._position(t); delta=t-self.times[i]
        return self.prefix[i]+self.rates[i]*delta+self.slopes[i]*delta**2/2

    def integrate(self,start,end):
        if end<start:
            raise ValueError('Reversed interval')
        value=self.primitive(end)-self.primitive(start)
        if np.any(value < -1e-12):
            raise ValueError('Negative exposure integral')
        # Only cancellation residuals within 1e-12 can be clipped to zero.
        return np.maximum(value,0)

    def interval_costs(self,start,dt,horizon):
        return np.asarray([self.integrate(start+k*dt,start+(k+1)*dt) for k in range(horizon)])


def nearest_exit_policy(graph):
    """Minimum move count, ties broken by the common action order."""
    distance=np.full(len(graph.cells),math.inf)
    queue=deque(sorted(graph.exit_ids))
    for i in queue:distance[i]=0
    while queue:
        j=queue.popleft()
        for _,i in graph.incoming[j]:
            if math.isinf(distance[i]):
                distance[i]=distance[j]+1;queue.append(i)
    policy=np.full(len(graph.cells),-1,dtype=int)
    for i in range(len(graph.cells)):
        if i in graph.exit_ids:continue
        if math.isfinite(distance[i]):
            policy[i]=min((a for a,j in graph.edges[i] if distance[j]==distance[i]-1))
    return policy,distance


def snapshot_policy(graph,current_rates):
    """Reversed Dijkstra: current rate frozen in time, with move-count tie break.

    Called anew after each observation; no future table rows are passed in.
    Constant dt cancels from the choice of the snapshot path.
    """
    rates=np.asarray(current_rates,dtype=float)
    if len(rates)!=len(graph.cells) or np.any(rates<0) or not np.isfinite(rates).all():
        raise ValueError('Nonnegative current node rates required')
    best=[(math.inf,math.inf)]*len(graph.cells)
    policy=np.full(len(graph.cells),-1,dtype=int)
    heap=[]
    for e in sorted(graph.exit_ids):
        best[e]=(0.,0);heapq.heappush(heap,(0.,0,e))
    while heap:
        dose,moves,j=heapq.heappop(heap)
        if (dose,moves)!=best[j]:continue
        for a,i in graph.incoming[j]:
            candidate=(rates[i]+dose,moves+1)
            if candidate<best[i]:
                best[i]=candidate;policy[i]=a
                heapq.heappush(heap,(*candidate,i))
            elif candidate==best[i] and (policy[i]<0 or a<policy[i]):
                policy[i]=a
    return policy


def future_min_fed_policy(graph,costs):
    """Backward dynamic programming, exact finite horizon to floating precision."""
    costs=np.asarray(costs,dtype=float)
    horizon,n=costs.shape
    if n!=len(graph.cells) or np.any(costs<0) or not np.isfinite(costs).all():
        raise ValueError('Finite nonnegative per-interval costs required')
    values=np.full((horizon+1,n),math.inf)
    moves=np.full((horizon+1,n),horizon+1,dtype=int)
    actions=np.full((horizon,n),-1,dtype=int)
    for e in graph.exit_ids:values[:,e]=0; moves[:,e]=0
    for k in range(horizon-1,-1,-1):
        for i in range(n):
            if i in graph.exit_ids:continue
            candidates=[(float(costs[k,i]+values[k+1,j]),1+int(moves[k+1,j]),a)
                        for a,j in graph.edges[i] if math.isfinite(values[k+1,j])]
            if candidates:
                values[k,i],moves[k,i],actions[k,i]=min(candidates)
    return actions,values,moves


def forward_min_fed_reference(graph,costs,start_id):
    """Independent forward shortest-path calculation on (step,cell) states."""
    horizon=costs.shape[0]
    best={(0,start_id):0.}
    queue=[(0.,0,start_id)]
    while queue:
        value,k,i=heapq.heappop(queue)
        if value!=best[(k,i)]:continue
        if i in graph.exit_ids:return value,k
        if k==horizon:continue
        for _,j in graph.edges[i]:
            candidate=value+float(costs[k,i]); key=(k+1,j)
            if candidate<best.get(key,math.inf):
                best[key]=candidate;heapq.heappush(queue,(candidate,k+1,j))
    return math.inf,None


def rollout(graph,actions,costs,start_id):
    """All policies share legal moves, residence accounting, horizon and terminal."""
    state=start_id; path=[graph.cells[state]]; trace=[];total=0.
    for k in range(costs.shape[0]):
        if state in graph.exit_ids:break
        a=int(actions[k,state]);destination=graph.destination(state,a)
        dose=float(costs[k,state]);total+=dose
        trace.append(dict(step=k,source=graph.cells[state],destination=graph.cells[destination],
                          action=a,FED_increment=dose,FED_cumulative=total))
        state=destination;path.append(graph.cells[state])
    return dict(reached=state in graph.exit_ids,steps=len(trace),FED=total,path=path,trace=trace,
                revisits=len(path)-len(set(path)),exit=graph.cells[state] if state in graph.exit_ids else None)
