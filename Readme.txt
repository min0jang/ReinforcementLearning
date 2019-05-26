Explanation and Demonstration:
https://minyoung.info/RL.html
https://www.youtube.com/watch?v=piQjevcBO6c&t=6s

You need to download carRL_env.py before running any of the codes, since it is the "environment" of the simulation

carRL.py : runs Reinforcement Learning, and stores trained model that meets the "requirement"
            - default requirement is to reach the flag with less than 100 penalty for 11 consecutive times.
            - this requirement can be easily modified

carRL_import.py : this is for running a simple test on trained model. After importing a trained model, you can 
                  input a state and see if the model outputs an action that makes sense
                  - for example, if your input state is : car is at bottom left with zero velocity and zero angular velocity
                          Then, a "well-trained" model would choose action #1, which is accelerate and steer left

carRL_run.py : this is for teting a trained model with the actual environment.
               - The difference between this one and carRL_import.py is that the user can run the rendering and 
               visually demonstrate that the model is working well.
