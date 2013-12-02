# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import pacman
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    ''' 
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py. 
    '''
    CaptureAgent.registerInitialState(self, gameState)

    ''' 
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    ''' 
    You should change this in your own agent.
    '''

    return random.choice(actions)


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    ''' 
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py. 
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.opponents = self.getOpponents(gameState)
    self.distributions = []
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    print self.legalPositions

    #initializing beleif distribution of opponents
    for i in range(0, gameState.getNumAgents()):
      if i in self.opponents:
        beliefs = util.Counter()
        for p in self.legalPositions: beliefs[p] = 1.0
        beliefs.normalize()
        self.distributions.append(beliefs)
      else:
        self.distributions.append(None)


    ''' 
    Your initialization code goes here, if you need any.
    '''

  def observe(self, agent, observation, gameState):
        """
        Updates beliefs based on the distance observation and Pacman's position.

        The noisyDistance is the estimated manhattan distance to the ghost you are tracking.

        """
        noisyDistance = observation
        pacmanPosition = gameState.getAgentPosition(self.index)

        allPossible = util.Counter()
        beliefs = self.distributions[agent]
        
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            emissionProb = gameState.getDistanceProb(trueDistance, noisyDistance) 
            if emissionProb > 0: 
                allPossible[p] = emissionProb * beliefs[p]

        "*** END YOUR CODE HERE ***"

        allPossible.normalize()
        self.distributions[agent] = allPossible

  def elapseTime(self, gameState):
    for o in self.opponents:
      allPossible = util.Counter()
      beliefs = self.distributions[o]

      for oldPos in self.legalPositions:
          successorStates = game.Actions.getLegalNeighbors(oldPos, gameState.getWalls())
          #print('successorState', successorStates)
          newPosDist = {}
          for pos in successorStates:
            prob = 1.0/len(successorStates)
            newPosDist[pos] = prob

          for newPos, prob in newPosDist.items():
              allPossible[newPos] += prob * beliefs[oldPos]

      allPossible.normalize()
      self.distributions[o] = allPossible

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    actions = gameState.getLegalActions(self.index)
    obs = gameState.getAgentDistances()
    for o in self.opponents:
      self.observe(o, obs[o], gameState)
    self.displayDistributionsOverPositions(self.distributions)

    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    #self.elapseTime(gameState)

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    #print('agent distances: ', gameState.getAgentDistances())

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

####################
# Inference module #
####################

import itertools
import util
import random
import capture
import pacman

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, agentIndex):
        "Sets the ghost agent for later access"
        self.agentIndex = agentIndex
        self.obs = [] # most recent observation position

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over
        ghost locations conditioned on all evidence so far.
        """
        pass

class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm
    updates to compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        """
        Updates beliefs based on the distance observation and Pacman's position.

        The noisyDistance is the estimated manhattan distance to the ghost you are tracking.

        """
        noisyDistance = observation
        pacmanPosition = gameState.getPacmanPosition()

        "*** YOUR CODE HERE ***"

        # Replace this code with a correct observation update
        # Be sure to handle the "jail" edge case where the ghost is eaten
        # and noisyDistance is None
        allPossible = util.Counter()
        
        for p in self.legalPositions:
            if noisyDistance is None:
                    allPossible[p] = 0.0
            else:
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                emissionProb = gameState.getDistanceProb(trueDistance, noisyDistance) 
                if emissionProb > 0: 
                    allPossible[p] = emissionProb * self.beliefs[p]

        "*** END YOUR CODE HERE ***"

        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        """
        Update self.beliefs in response to a time step passing from the current state.

        The transition model is not entirely stationary: it may depend on Pacman's
        current position (e.g., for DirectionalGhost).  However, this is not a problem,
        as Pacman's current position is known.

        In order to obtain the distribution over new positions for the
        ghost, given its previous position (oldPos) as well as Pacman's
        current position, use this line of code:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        Note that you may need to replace "oldPos" with the correct name
        of the variable that you have used to refer to the previous ghost
        position for which you are computing this distribution. You will need to compute
        multiple position distributions for a single update.

        newPosDist is a util.Counter object, where for each position p in self.legalPositions,

        newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos at time t )

        (and also given Pacman's current position).  You may also find it useful to loop over key, value pairs
        in newPosDist, like:

          for newPos, prob in newPosDist.items():
            ...

        *** GORY DETAIL AHEAD ***

        As an implementation detail (with which you need not concern
        yourself), the line of code at the top of this comment block for obtaining newPosDist makes
        use of two helper methods provided in InferenceModule above:

          1) self.setGhostPosition(gameState, ghostPosition)
              This method alters the gameState by placing the ghost we're tracking
              in a particular position.  This altered gameState can be used to query
              what the ghost would do in this position.

          2) self.getPositionDistribution(gameState)
              This method uses the ghost agent to determine what positions the ghost
              will move to from the provided gameState.  The ghost must be placed
              in the gameState with a call to self.setGhostPosition above.

        It is worthwhile, however, to understand why these two helper methods are used and how they
        combine to give us a belief distribution over new positions after a time update from a particular position
        """

        "*** YOUR CODE HERE ***"

        allPossible = util.Counter()

        for oldPos in self.legalPositions:
            actions = gameState.getLegalActions(agentIndex)
            successorStates = [gameState.generateSuccessor(action) for action in actions]
            newPosDist = {}
            for state in successorStates:
              position = state.getAgentPosition(agentIndex)
              prob = 1.0/len(actions)
              newPosDist[position] = prob

            for newPos, prob in newPosDist.items():
                allPossible[newPos] += prob * self.beliefs[oldPos]

        allPossible.normalize()
        self.beliefs = allPossible

    def getBeliefDistribution(self):
        return self.beliefs
