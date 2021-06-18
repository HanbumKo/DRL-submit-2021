# Find the worst case using Deep Reinforcement Learning

### Description

Train **instance generating agent(g_agent)** that TSP problem solving agent(s_agent) can’t cover.





### Dependencies

Ubuntu 18.04

PyTorch

matplotlib





### Run training

```bash
python main.py
```





### Code description

`/DFPG_A2C_hybrid` : s_agent model weight file **(deleted due to private project)**

`config.py` : configurations, options

`env.py` : TSP environment, Generator environment **(TSP env is deleted due to private project)**

`graph_encoder.py` : graph encoder model implementation **(deleted due to private project)**

`main_generate_instance.py` : run file, generate instances using trained g_agent.

`main_random_action.py` : run file, test random agent

`nnets` : NN model implementation(used for s_agent) **(deleted due to private project)**

`plot.py` : run file, plot instance example

`plot_result_DDPG.py` : run file, plot DDPG result

`plot_result_REINFORCE.py` : run file, plot REINFORCE result

`reinforce_nnets.py` : REINFORCE NN implementation

`reaplay_buffer.py` : DDPG replay buffer implementation





### Training Result

![REINFORCE_result](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/REINFORCE_test_reward.png?raw=true)



### Generated Instances

![Instance0](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_0.png?raw=true)

![Instance1](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_1.png?raw=true)

![Instance2](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_2.png?raw=true)

![Instance3](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_3.png?raw=true)

![Instance4](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_4.png?raw=true)

![Instance5](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_5.png?raw=true)

![Instance6](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_6.png?raw=true)

![Instance7](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_7.png?raw=true)

![Instance8](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_8.png?raw=true)

![Instance9](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_9.png?raw=true)

![Instance10](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_10.png?raw=true)

![Instance11](https://github.com/HanbumKo/DRL-submit-2021/blob/20215015-hanbum/README_images/generated_instance/instance_11.png?raw=true)

