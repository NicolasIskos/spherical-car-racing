# Setup

In the diagrams below, the car starts from (X=10, Y=0), drives around a cone at position (X=20, Y=10), and crosses the finish line at X=80. It simulates a simple Autocross corner.

## Environment

The state space is descretized in both spatial and velocity dimensions. At any given state, the available next states are dictated by the traction of the tires (the car is assumed to have infinite power). If the car hits a wall, it restarts at the beginning of the track. The episode only ends when the car crosses the finish line.

## Reinforcement Learning Setup

We aim to use two different reinforcement learning techniques. In each case,
we define a value function $Q(s,a)$ where $s$ is the state and $a$ is the action taken at that state. We also define a reward $r$ for each possible state-action pair. In our case, $r$ wil always be -1. This way, the highest total reward will come when the number of actions (and thus time steps) is minimized.

## e-Greedy Monte Carlo

The first approach is e-Greedy Monte Carlo. The car greedily chooses the action $a$ that has the highest $Q(s, a)$ with probability 1-epsilon. Otherwise, the car chooses randomly. At the end of the episode, all traversed states are updated according to the following relation (on first-visit).

$Q(s,a) = \gamma Q(s',a') + r$

where where $s$ and $a$ are the state and action that immediately precede $s'$ and $a'$. $\gamma$ is a discount factor typically in the range of 0.9 to 1.

Now we apply this technique to the racetrack problem.

Note: For the purposes of visualization, all possible velocity state values are averaged for each position and color coded based numerical value. The goal is for the highest value states to trace out the path of an optimal racing line.

![Monte Carlo](images/monte_carlo.png)
- Solution is often way off optimal
- Often converges to a value worse than optimal
- Sensitive to initial conditions
- Setting initial Q too small means policy rarely explores new states if there’s an option to go to a known state -> policy is suboptimal 
- Setting initial Q too large means policy frequently explores new states when there’s an option to go to an unknown state -> policy converges very slowly, doesn’t care even when a decent solution is found.

## SARSA Semi-gradient

The second technique is SARSA Semi-gradient. Like Monte Carlo, SARSA uses an epsilon-greedy approach for choosing the next state. But rather than updating states only at the end of the episode, this technique updates states as it traverses them. Specifically,

$Q(s,a) = Q(s,a) + \alpha[r+\gamma Q(s', a') - Q(s, a)]$

where $\alpha$ is the learning rate.

This continuous state update based on the values of proximal states is what makes SARSA a temporal difference model. It is kind of a hybrid between dynamic programming and Monte Carlo techniques because it updates based on the values of proximal states, like DP, but does so following the path of randomly sampled episodes, like MC.

A few notes on applying this technique to the racetrack problem:

![Sarsa Semi-gradient](images/sarsa.png)
- Works significantly better than MC.
- Converges faster and to a better solution.
- Comparing state diagrams, MC explores alternate paths unevenly.
SARSA explores alternate paths evenly.
- When an MC alternate fails to get to the end, earlier parts of the path are heavily negatively reinforced because they occurred early and we wait until the end of the episode to apply updates.
- When an MC alternate succeeds in getting to the end, it gets heavily positively reinforced because the state values of that alternate only improve - keeps taking the same actions on subsequent episodes.
- This leads to high-variance, inaccurate estimations of alternates.
- In SARSA, alternate path states are updated the same way regardless of failure or success, until success state values from the end of the track propagate backward.
- This means state values are not estimated prematurely.
- Because SARSA updates throughout the episode, alternate paths don’t get heavily traversed run-after-run until (well-known) success state values propagate backward.
