# Predicting Customer Behaviours Using Adversarial Imitation Learning

## Follow these steps to install:
1. Clone folder
2. Create virtual environment, e.g., python3 -m venv venv
3. Activate virtual environment, run source venv/bin/activate
4. Run pip install -r requirements.txt
5. Go to customer_behaviour/chainerrl and run pip install -e .
6. Go to customer_behaviour/custom_gym and run pip install -e .
7. Run example by running python main.py mmct-gail --state CS

Note: The first input, i.e., mmct-gail, denotes which algorithm that is running whereas the state parameter denotes the chosen state representation. There are three available algorithms (gail, airl and mmct-gail) and three state representations (basic, one-hot and CS). 
