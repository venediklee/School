import numpy as np
def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """
    N = A.shape[0]
    M = B.shape[1]
    T = O.shape[0]
    probabilities = np.zeros((N,T))
    #initilalize the probabilities at T=0 with initial state probabilities
    for stateID in range(N):
        probabilities[stateID][0] = pi[stateID] * B[stateID][O[0]]

    #for each timestep t, 1<=t<T calculate state probabilities
    for t in range(1,T):
        #for each stateID in [0,N] calculate probability of getting to that state by
        # multiplying previous state's probability and transition probability
        for stateID in range(N):
            probability = 0
            for previousStateID in range(N):#calculate the probability of moving from prev state to this state
                probability+=probabilities[previousStateID][t - 1] * A[previousStateID][stateID]
            #multiply probability with observing this state at time t
            probability*=B[stateID][O[t]]
            probabilities[stateID][t] = probability

    observationProbability = 0
    for stateID in range(N):
        observationProbability+=probabilities[stateID][T - 1]

    return observationProbability,probabilities

def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    N = A.shape[0]
    M = B.shape[1]
    T = O.shape[0]
    probabilities = np.zeros((N,T))
    #initilalize the probabilities at T=0 with initial state probabilities
    for stateID in range(N):
        probabilities[stateID][0] = pi[stateID] * B[stateID][O[0]]
    
    #for each timestep t, 1<=t<T calculate state probabilities
    for t in range(1,T):
        #for each stateID in [0,N] calculate probability of getting to that state by
        # finding the max probability of
        #           multiplying previous state's probability and transition probability
        # and then multiplying it with the probability of seeing the observation at t in this state
        for stateID in range(N):
            maxProbability = 0
            maxProbabilityStateID = 0
            for previousStateID in range(N):#calculate the MAX probability of moving from prev state to this state
                probability = probabilities[previousStateID][t - 1] * A[previousStateID][stateID]
                if(probability > maxProbability):
                    maxProbability = probability
                    maxProbabilityStateID = previousStateID
            #multiply probability with observing this state at time t
            maxProbability*=B[stateID][O[t]]
            probabilities[stateID][t] = maxProbability
    highestProbabilitySequence = np.zeros(T,dtype=int)
    for t in reversed(range(T)):
        highestProbabilitySequenceValue = 0
        for stateID in range(N):
            if(probabilities[stateID][t] > highestProbabilitySequenceValue):
                if(t == T - 1 or (A[highestProbabilitySequence[t]][highestProbabilitySequence[t + 1]] > 0)):
                    highestProbabilitySequenceValue = probabilities[stateID][t]
                    highestProbabilitySequence[t] = stateID
    return highestProbabilitySequence,probabilities