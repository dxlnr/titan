"""Training Procedure"""
import torch


class MuZeroLoss:
    """MuZero Loss Function."""
    
    def __init__(self, config: Conf):
        if config.LOSS = "mse": 
            self.loss = nn.MSELoss()
        else:
            raise Exception(f"{config.LOSS} loss not implemented.")

    def __call__(v, r, p, t_v, t_r, t_p):
        v_loss = loss(v, t_v)
        r_loss = loss(r, t_r)
        p_loss = loss(p, t_p)
        return v_loss, r_loss, p_loss


def train(config: Conf, store: SharedStorage, replay_buffer: ReplayBuffer):
    """."""
    model = build_model(config)
    model.train()
    model.to(torch.device("cuda" if torch.cuda_available() else "cpu"))
    
    # lr = config.LR_INIT * config.LR_DECAY_RATE **(tf.train.get_global_step() / config.lr_decay_steps)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LR_INIT, weight_decay=config.WEIGHT_DECAY)

    for i in range(config.TRAINING_STEPS):
        if i % config.CHECKPOINT_INTERVAL == 0:
            store.save_model(i, model)

        batch = replay_buffer.sample_batch(config.NUM_UNROLL_STEPS, config.TD_STEPS)
        loss_function = MuZeroLoss(config)
        loss = 0

        for obs, actions, targets in batch:
            # # Initial step, from the real observation.
            v, r, p, s = model.initial_inference(obs)
            predictions = [(1.0, v, r, p)]

            # # Recurrent steps, from action and previous hidden state.
            for a in actions:
                v, r, p, s = model.recurrent_inference(s, a)
            # predictions.append((1.0 / len(actions), value, reward, policy_logits))

            # # hidden_state = tf.scale_gradient(hidden_state, 0.5)

            # for prediction, target in zip(predictions, targets):
            #     gradient_scale, value, reward, policy_logits = prediction
            #     target_value, target_reward, target_policy = target

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.training_step += 1

        # l = (scalar_loss(value, target_value) + scalar_loss(reward, target_reward) + tf.nn.softmax_cross_entropy_with_logits(
        #       logits=policy_logits, labels=target_policy))

        # loss += tf.scale_gradient(l, gradient_scale)

    # for weights in network.get_weights():
    #     loss += weight_decay * tf.nn.l2_loss(weights)

    # optimizer.minimize(loss)


    store.save_model(config.TRAINING_STEPS, model)


