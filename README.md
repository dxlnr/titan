<h1 align="center">
  <b>TITAN</b><br>
</h1>


Zero Knowledge Chess Engine


### MU Zero 
policy and value networks

- The **policy network** is a probability distribution over all actions *a* that can be taken in state *s*.
    It enables the system to estimate the actions that are most likely to lead to success.

- The **value network** estimates the probability of winning from the current state *s*.
    As the action space can get very large, this network allows for cutting down the search 
    tree as it allows for looking ahead only a few moves and then rely on the network to 
    evaluate the current state.

- The Monte-Carlo Tree Search gives a schematic structure on how to choose these estimates from 
    both the policy and the value network. The algorithm allows for evaluation of many 
    interesting actions and aggregate the information for better action selection. (Exploration vs Exploitation)

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
