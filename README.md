# An End-to-End Reinforcement Learning Framework for Autonomous Navigation without Prior Knowledge

---

### Abstract
  >Real-world autonomous navigation demands the ability to operate in unknown, unstructured environments 
  using only onboard, limited-range sensor data, without reliance on prior maps or global information. 
  To address this fundamental challenge, this paper introduces a novel end-to-end navigation framework 
  that seamlessly integrates perception and planning. The core of our framework is Perception Net, an 
  interpretable, optimization-unfolded network that efficiently processes raw point clouds into a 
  compact representation of the immediate environment. This is coupled with a reinforcement learning 
  planner featuring a velocity-adaptive Bézier action space, which learns a sophisticated, reference-free 
  navigation policy. The agent's decisions are based solely on its own velocity, the relative goal 
  coordinates, and the real-time perception features. Extensive simulations demonstrate the framework's 
  high success rate and robust generalization to unseen non-convex scenarios. The practicality and 
  effectiveness of this map-less, purely reactive system are further validated through successful 
  real-world field tests on an Autonomous Surface Vehicle (ASV) platform, showcasing its capability for 
  truly autonomous navigation in complex environments. Future work will focus on handling dynamic 
  obstacles and improving robustness in escaping from extremely non-convex traps.

---

### Acknowledgments & License

This project is built upon several excellent open-source projects. We extend our sincere gratitude to their developers.

*   **[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3):** Our reinforcement learning agent is implemented using the robust and versatile Stable Baselines3 library.

*   **[NeuPAN](https://github.com/hanruihua/NeuPAN):** Our implementation and comparative analysis heavily reference and include modified code from the official NeuPAN repository. We thank the original authors for making their work public.

**License:** The original NeuPAN project is licensed under the **GNU General Public License v3.0**. In accordance with its terms, any derivative work must also be distributed under the same license. Therefore, this repository and all its contents are also licensed under the **GNU General Public License v3.0**. Please see the `LICENSE` file for more details.