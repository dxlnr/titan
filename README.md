<h1 align="center">
  <b>TITAN</b><br>
</h1>


Zero Knowledge Chess Engine


### MU Zero 
policy and value networks

- The policy, written p(s,a), is a probability distribution over all actions aa that can be taken in state s.
- The value v(s) estimates the probability of winning from the current state s.


TODO:
- Adjust the UCB formula for MuZero
- Add policy & value network (actually neural networks)
- Add a training function for the whole thing
- Connect the move generation function to the interface
- Let the thing beat you (lol)


### Monte Carlo Tree Search

1. Tree traversel 
2. Node expansion
3. Rollout
4. Backpropagation

Should be pretty much in place.
