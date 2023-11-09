import HMM

print("Running Question 2")
model = HMM.HMM()

print("Running Generate function with n = 20")
model.load("partofspeech.browntags.trained")
observation = model.generate(20)
print(observation)

print("\n\n Running Viterbi function")
model = HMM.HMM()
model.load("partofspeech.browntags.trained")
with open("ambiguous_sents.obs", "r") as f:
    lines = f.readlines()
    for line in lines:
        words = line.strip().split()
        observation = HMM.Observation([""] * len(words), words)
        best_path = model.viterbi(observation)
        print("Most likely sequence of states:", best_path)

print("\n\n Running Forward function")
model = HMM.HMM()
model.load("partofspeech.browntags.trained")
with open("ambiguous_sents.obs", "r") as f:
    lines = f.readlines()
    for line in lines:
        words = line.strip().split()
        observation = HMM.Observation([""] * len(words), words)
        final_state, final_prob = model.forward(observation)
        print("Most likely final state:", final_state)
        print("Probability of the observation:", final_prob)


print("\n\n\nRunning Question 3")
print("Running alarm.py")
# detect python3 or python2
import sys

if sys.version_info[0] < 3:
    execfile("alarm.py")
else:
    exec(open("alarm.py").read())


print("\n\nRunning carnet.py")
if sys.version_info[0] < 3:
    execfile("alarm.py")
else:
    exec(open("carnet.py").read())
