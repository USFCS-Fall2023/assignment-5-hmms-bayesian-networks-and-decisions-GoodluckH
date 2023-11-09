import random
import argparse
import codecs
import os
import numpy


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return " ".join(self.stateseq) + "\n" + " ".join(self.outputseq) + "\n"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions
        self.state_index = {}  # New attribute for state to index mapping

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        trans_file = basename + ".trans"
        emit_file = basename + ".emit"

        transitions = {}
        emissions = {}

        with open(trans_file, "r") as f_trans:
            for line in f_trans:
                line = line.split()
                if len(line) == 3:
                    from_state, to_state, prob = line
                    prob = float(prob)  # convert to float
                    if from_state not in transitions:
                        transitions[from_state] = {}
                    transitions[from_state][to_state] = prob

        with open(emit_file, "r") as f_emit:
            for line in f_emit:
                line = line.split()
                if len(line) == 3:
                    state, output, prob = line
                    prob = float(prob)
                    if state not in emissions:
                        emissions[state] = {}
                    emissions[state][output] = prob

        self.transitions = transitions.copy()
        self.emissions = emissions.copy()
        self.state_index = {state: i for i, state in enumerate(self.transitions.keys())}

    ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        if not self.transitions or not self.emissions:
            raise ValueError("HMM model is not properly initialized")

        # Initialize sequences for states and outputs
        state_sequence = []
        output_sequence = []

        # Start with the initial state (assumed to be '#')
        current_state = "#"

        for _ in range(n):
            # Sample the next state based on transition probabilities
            next_state = random.choices(
                list(self.transitions[current_state].keys()),
                weights=self.transitions[current_state].values(),
            )[0]

            # Sample the output based on emission probabilities
            output = random.choices(
                list(self.emissions[next_state].keys()),
                weights=self.emissions[next_state].values(),
            )[0]

            # Append the state and output to their respective sequences
            state_sequence.append(next_state)
            output_sequence.append(output)

            # Update the current state for the next iteration
            current_state = next_state

        return Observation(state_sequence, output_sequence)

    def forward(self, obs):
        # Guard clause for empty observation sequence
        if not obs.outputseq:
            return None, 0

        # Initialize forward matrix
        num_states = len(self.transitions)
        forward_matrix = numpy.zeros((len(obs), num_states))

        # Assign initial probabilities
        init_state = "#"
        for state, idx in self.state_index.items():
            if state == init_state:
                continue
            forward_matrix[0][idx] = self.transitions[init_state][
                state
            ] * self.emissions.get(state, {}).get(obs.outputseq[0], 0)

        # Compute forward probabilities
        for time_step in range(1, len(obs)):
            for curr_state, curr_idx in self.state_index.items():
                if curr_state == init_state:
                    continue
                forward_matrix[time_step][curr_idx] = self.calculate_forward_prob(
                    time_step, curr_state, curr_idx, obs, forward_matrix
                )

        # Total probability of observation sequence
        total_prob = sum(forward_matrix[-1])
        return forward_matrix, total_prob

    def calculate_forward_prob(self, t, curr_state, curr_idx, obs, fwd_matrix):
        """Calculate forward probability for a given state and time."""
        return sum(
            fwd_matrix[t - 1][prev_idx]
            * self.transitions[prev_state].get(curr_state, 0)
            * self.emissions.get(curr_state, {}).get(obs.outputseq[t], 0)
            for prev_state, prev_idx in self.state_index.items()
            if prev_state != "#"
        )

    def viterbi(self, observation):
        # Return None if observation is empty
        if len(observation.outputseq) == 0:
            return None

        # Create a mapping from state labels to numeric indices
        state_list = list(self.transitions)
        index_map = {state: index for index, state in enumerate(state_list)}

        # Set up the arrays for Viterbi probabilities and backtrace pointers
        num_states = len(state_list)
        prob_matrix = numpy.zeros((len(observation), num_states))
        path_matrix = numpy.zeros((len(observation), num_states), dtype=int)

        # Initialize the probability matrix
        for state in state_list:
            idx = index_map[state]
            prob_matrix[0][idx] = (
                self.emissions[state][observation.outputseq[0]]
                if state in self.emissions
                and observation.outputseq[0] in self.emissions[state]
                else 0.0
            )

        # Compute Viterbi probabilities for each state and time step
        for time_step in range(1, len(observation)):
            for state in state_list:
                state_idx = index_map[state]
                if state in self.emissions:
                    transition_probs = [
                        prob_matrix[time_step - 1][index_map[prev_state]]
                        * self.transitions[prev_state].get(state, 0)
                        * self.emissions[state].get(observation.outputseq[time_step], 0)
                        for prev_state in state_list
                    ]
                    if transition_probs:
                        max_prob = max(transition_probs)
                        prob_matrix[time_step][state_idx] = max_prob
                        path_matrix[time_step][state_idx] = transition_probs.index(
                            max_prob
                        )

        # Determine the most probable final state
        final_state = numpy.argmax(prob_matrix[-1])

        # Backtrack to find the most probable path
        optimal_path = [final_state]
        for time_step in reversed(range(1, len(observation))):
            optimal_path.insert(0, path_matrix[time_step][optimal_path[0]])

        # Map numeric indices back to state labels
        optimal_state_sequence = [state_list[idx] for idx in optimal_path]
        return optimal_state_sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM")
    parser.add_argument("filename", type=str, help="filename to generate")
    # parser.add_argument("--generate", type=int, help="generate random n observations")
    parser.add_argument(
        "--viterbi", type=str, help="run viterbi algorithm on an observation"
    )
    # parser.add_argument(
    #     "--forward", type=str, help="run forward algorithm on an observation"
    # )

    args = parser.parse_args()

    model = HMM()

    model.load(args.filename)

    # if args.generate > 0:
    #     # Generate n observations
    #     for _ in range(args.generate):
    #         observation = model.generate(20)  # Modify the number 20 as needed
    #         print(observation)

    if args.viterbi:
        # Run Viterbi on the specified input observation file
        with open(args.viterbi, "r") as f:
            lines = f.readlines()
            for line in lines:
                words = line.strip().split()
                observation = Observation([""] * len(words), words)
                best_path = model.viterbi(observation)
                print("Most likely sequence of states:", best_path)

    # if args.forward:
    #     # Run forward algorithm on the specified input observation file
    #     with open(args.forward, "r") as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             words = line.strip().split()
    #             observation = Observation([""] * len(words), words)
    #             final_state, final_prob = model.forward(observation)
    #             print("Most likely final state:", final_state)
    #             print("Probability of the observation:", final_prob)
