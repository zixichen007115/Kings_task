import numpy as np
import pandas as pd

# parameters
N_ROW = 4
N_COL = 12
N_STA = N_ROW * N_COL
ACTIONS = ['l','r','u','d']
E = 0.1
ALPHA = 0.1
GAMMA = 1
MAX_TRAN = 50
EPISO = 500

# initialize q table
def q_table_in(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns = actions,
    )
    # boundary
    table['d'][:12]=np.nan
    table['u'][36:]=np.nan
    table['l'][0,12,24,36]=np.nan
    table['r'][11,23,35,47]=np.nan
    return table

# select available action
def act(state, q_table):
    actions = q_table.loc[state, :]
    # boundary
    if state < 12:
        actions = actions.drop('d', axis=0, inplace=False)
    if state > 35:
        actions = actions.drop('u', axis=0, inplace=False)
    if state % 12 == 0:
        actions = actions.drop('l', axis=0, inplace=False)
    if state % 12 == 11:
        actions = actions.drop('r', axis=0, inplace=False)

    # random  or max q selection
    if np.random.uniform()<E or actions.all() == 0:
        action = actions.sample(n=1, axis=0)
        action = action.index[0]
    else:
        action = actions.idxmax()
    return action

# calculate future state, reward and flag after action
def feedback(cur_state, action):
    reward = -1
    fur_state = cur_state
    flag = 'fail'

    # update state
    if action == 'u':
        fur_state += 12
    elif action == 'd':
        fur_state -= 12
    elif action == 'r':
        fur_state += 1
    elif action == 'l':
        fur_state -= 1

    # cliff or terminal
    if fur_state > 0 and fur_state <11:
        fur_state = 0
        reward +=-100
    elif fur_state == 11:
        flag = 'succeed'
    return fur_state, reward, flag

# sarsa process
def sarsa(q_table):
    # start point
    state = 0
    sum_reward = 0
    action = act(state, q_table)
    flag='fail'

    # walking
    for i in range(MAX_TRAN):
        if flag != 'succeed':

            # state calculation
            fur_state, reward, flag = feedback(state, action)

            # q table update
            q=q_table.loc[state, action]
            if flag == 'succeed':
                q += ALPHA * (reward -q)
            else:
                # act calculation
                fur_action = act(fur_state, q_table)
                q += ALPHA * (reward + GAMMA*q_table.loc[fur_state,fur_action] -q)
            q_table.loc[state, action] = q

            # reward, action and state update
            sum_reward += reward
            action = fur_action
            state = fur_state
    return i, sum_reward, flag, q_table

if __name__ == '__main__':
    reward_list = np.zeros(EPISO)
    for i in range(EPISO):
        if i == 0:
            q_table = q_table_in(N_STA, ACTIONS)
        step, sum_reward, final_state, q_table = sarsa(q_table)
        if i%10 ==9:
            print('Eposode {}, final state:{}, reward sum:{}.'.format(i, final_state, sum_reward))
        reward_list[i]=sum_reward
    np.save('sarsa.npy',reward_list)

