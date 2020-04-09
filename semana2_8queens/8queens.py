import math
import random
class Queen():
    row=0
    column=0
    def __init__(self,row,column):
        self.row=row
        self.column=column
    def move(self):
        self.row=self.row+1
    def ifConflict(self,queen):
        #check row and columns
        if self.row==queen.getRow() or self.column==queen.getColumn():
            return True
        elif math.fabs(self.column-queen.getColumn()) == math.fabs(self.row-queen.getRow()):
            return True
        return False
    def getRow(self):
        return self.row
    def getColumn(self):
        return self.column
class HillClimbingRandomRestart():
    n=4
    stepsClimbedAfterLastRestart=0
    stepsClimbed=0
    heuristic=0
    randomRestarts=0
    
    def printState(self,state):
        tempBoard=[[0 for j in range(self.n)] for i in range(self.n)]
        for i in range(self.n):
            tempBoard[state[i].getRow()][state[i].getColumn()]=1
        for i in range(self.n):
            for j in range(self.n):
                print(tempBoard[i][j]," ")
            print("\n")
    def generateBoard(self):
        startBoard=[None]*self.n
        for i in range(self.n):
            startBoard[i]=Queen(random.randint(0,self.n-1),i)
        return startBoard
    def findHeuristic(self,state):
        heuristic=0
        for i in range(self.n):
            for j in range(self.n):
                if state[i].ifConflict(state[j]):
                    heuristic+=1
                    
        return heuristic
    def nextBoard (self,presentBoard):
        nextBoard=[None]*self.n
        tmpBoard=[None]*self.n
        presentHeuristic=self.findHeuristic(presentBoard)
        bestHeuristic=presentHeuristic
        tempH=0
        for i in range(self.n):
            nextBoard[i]=Queen(presentBoard[i].getRow(), presentBoard[i].getColumn())
            tmpBoard[i]=Queen(presentBoard[i].getRow(), presentBoard[i].getColumn())
        for i in range(self.n):
            if i>0:
                tmpBoard[i-1]=Queen(presentBoard[i-1].getRow(),presentBoard[i-1].getColumn())
            tmpBoard[i]=Queen(0,tmpBoard[i].getColumn())
            
            for j in range(self.n):
                tempH=self.findHeuristic(tmpBoard)
                if tempH<bestHeuristic:
                    bestHeuristic=tempH
                    for k in range(self.n):
                        nextBoard[k]=Queen(tmpBoard[k].getRow(),tmpBoard[k].getColumn())
                    
                if tmpBoard[i].getRow()!=self.n-1:
                    tmpBoard[i].move()
        if bestHeuristic==presentHeuristic:
            self.randomRestarts+=1
            self.stepsClimbedAfterLastRestart=0
            nextBoard=self.generateBoard()
            self.heuristic=self.findHeuristic(nextBoard)
        else:
            self.heuristic=bestHeuristic
        self.stepsClimbed+=1
        self.stepsClimbedAfterLastRestart+=1
        return nextBoard
    def main(self):
        presentHeuristic=0
        while True:
            if self.n==2 or self.n==3:
                print("No solution possible for that entry")
            else:
                break
        presentBoard=self.generateBoard()
        presentHeuristic=self.findHeuristic(presentBoard)
        
        while presentHeuristic !=0:
            presentBoard=self.nextBoard(presentBoard)
            presentHeuristic=self.heuristic
        self.printState(presentBoard)
            
                        
def init():
    HillClimbingRandomRestart().main()
    
init()