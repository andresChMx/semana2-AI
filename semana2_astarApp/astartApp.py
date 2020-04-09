#conda install kivy -c conda-forge
import math

from kivy.config import Config

from kivy.app import App

from kivy.uix.widget import Widget

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

from kivy.lang import Builder

Config.set('graphics', 'width', '920')
Config.set('graphics', 'height', '600')

Builder.load_string("""
<Tile>:
    dimension: 19,19
    pos:0,0
    color:(0,1,0,1)
    canvas:
        PushMatrix
        Color:
            rgba: self.color
        Rectangle:
            pos:self.pos
            size: self.dimension    
        PopMatrix                    
<Canvas>:
    dimension: 800,500
    pos:60,60
    canvas:
        PushMatrix
        Color:
            rgba: 1,1,1,1
        Rectangle:
            pos:self.pos
            size: self.dimension
        PopMatrix      
""")
world_width=800
world_height=500
world_posx=60
world_posy=60
tile_size=20

list_tiles=[]


class Tile(Widget):
    start=None
    goal=None
    type_of_tile=0
    def __init__(self,x,y,**kwargs):
        super(Tile,self).__init__(**kwargs)
        self.pos=(x,y)
        self.hs=(self.getPos()[0]-self.goal[0])**2+(self.getPos()[1]-self.goal[1])**2
        self.fs=0
        self.parent_node=None
    def isGoal(self):
        return self.getPos()==self.goal

    def set_type(self,i):

        if i==0:
            self.color=(0,1,0,1)
            self.type_of_tile=i
        elif i==1:
            self.color=(0.6,0.29,0.16,1.0)
            self.type_of_tile=i        
        elif i==2:
            self.color=(0.7,0.7,0.7,1)
            self.type_of_tile=i
        elif i==3:
            self.color=(1,0.73,0,1)
            self.type_of_tile=i
        else:
            self.color=(1,0.3,0.3,1)        
    def getPos(self):
        fixedX=self.pos[0]-world_posx
        fixedY=self.pos[1]-world_posy
        indexI=math.floor(fixedY/tile_size)
        indexJ=math.floor(fixedX/tile_size)
        return (indexJ,indexI)
            
class EventManager(Widget):
    parent_tiles_list=None
    def on_touch_down(self, touch):
        pass

    def on_touch_move(self, touch):
        fixedX=touch.x-world_posx
        fixedY=touch.y-world_posy
        indexI=math.floor(fixedY/tile_size)
        indexJ=math.floor(fixedX/tile_size)
        if (indexJ<0 or indexJ>=(world_width/tile_size)) or (indexI<0 or indexI>=world_height/tile_size):
            pass
            #print("fuera de rango")
        else:
            tmp_index_1d=math.floor(indexI*int(world_width/20))+indexJ
            self.parent_tiles_list[tmp_index_1d].set_type(1)

class Canvas(Widget):
    pass
class GameApp(App):
    def build(self):
        parent=Canvas()
        eventManager=EventManager()
        btnStart=Button(text="start",size=(100,50))
        btnRestart=Button(text="restart",pos=(100,0),size=(100,50))
        
        btnRestart.bind(on_release=self.btn_restart)
        btnStart.bind(on_release=self.btn_start)
        
        parent.add_widget(eventManager)
        parent.add_widget(btnStart)
        parent.add_widget(btnRestart)
        
        Tile.start=(0,0)
        Tile.goal=(39,24)
    
        for i in range(int(world_height/20)):
            for j in range(int(world_width/20)):
                tmp=Tile((j*20)+parent.pos[0],(i*20)+parent.pos[1])
                parent.add_widget(tmp)
                list_tiles.append(tmp)

        eventManager.parent_tiles_list=list_tiles
        
        list_tiles[indexesIJ2index(Tile.start[1], Tile.start[0])].set_type(2)
        list_tiles[indexesIJ2index(Tile.goal[1], Tile.goal[0])].set_type(3)

        return parent
    def btn_start(self,obj):
        astart()
    def btn_restart(self,obj):
        for x in list_tiles:
            x.set_type(0)
        list_tiles[0].set_type(2)
        list_tiles[999].set_type(3)

    
def indexesIJ2index(i,j):
    return math.floor(i*int(world_width/20)+j)

class NodeList(list):
    def find(self,x,y):
        l=[i for i in self if i.getPos()==(x,y)]
        return l[0] if l!=[] else None
    def remove(self,node):
        del self[self.index(node)]

# press hs been press
        
def astart():

    open_list=NodeList()
    close_list=NodeList()
    start_node=list_tiles[indexesIJ2index(Tile.start[1],Tile.start[0])]
    start_node.fs=start_node.hs
    open_list.append(start_node)
    
    final_node=None
    
    while True:
        if open_list==[]:
            print("There are not routes until reaching the goal")
            break
        
        n=min(open_list,key=lambda x:x.fs)
        open_list.remove(n)
        close_list.append(n)
        if n.isGoal():
            final_node=n
            break
        
        n_gs=n.fs-n.hs
        for x in ((1,0),(0,-1),(-1,0),(0,1)):
            new_x=n.getPos()[0]+x[0]
            new_y=n.getPos()[1]+x[1]
            
            if not ((new_x>=0 and new_x<int(world_width/20)) and (new_y>=0 and new_y<int(world_height/20)) and list_tiles[indexesIJ2index(new_y,new_x)].type_of_tile!=1):
                continue
            
            m=open_list.find(new_x,new_y)
            dist=(n.getPos()[0]-new_x)**2+(n.getPos()[1]-new_y)**2
            if m:
                if m.fs > n_gs + m.hs + dist:
                    m.fs = n_gs + m.hs + dist
                    m.parent_node = n
            else:
                m=close_list.find(new_x,new_y)
                if m:
                    if m.fs > n_gs + m.hs + dist:
                        m.fs = n_gs + m.hs + dist
                        m.parent_node = n
                        open_list.append(m)
                        close_list.remove(m) 
                else:
                    m=list_tiles[indexesIJ2index(new_y,new_x)]
                    m.parent_node=n
                    m.fs=m.hs+n_gs+dist
                    open_list.append(m)
    
    if final_node!=None:
        while True:
            if final_node.parent_node==None:
                break
            final_node.set_type(4)
            final_node=final_node.parent_node

GameApp().run()