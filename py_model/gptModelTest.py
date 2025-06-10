import random

def generate_recurrent_data(num_cases, sequence_length, input_size, output_range=(-1, 1)):
    cases = []
    answers = []
    for _ in range(num_cases):
        case = []
        for _ in range(sequence_length):
            inputs = [random.uniform(-1, 1) for _ in range(input_size)]
            case.append(inputs)
        cases.append(case)

        # Example: target could be average of all inputs, clipped to output_range
        total = sum(sum(inputs) for inputs in case)
        target = max(output_range[0], min(output_range[1], total / (sequence_length * input_size)))
        answers.append([target])

    return cases, answers

# Example usage
teach_cases, teach_answers = generate_recurrent_data(num_cases=100, sequence_length=5, input_size=1)



from Model import RecurrentLearning, RecurrentModel

model = RecurrentModel(inputs=1, hiddenLayerNodes=3, hiddenLayers=1, outputs=1)
learning = RecurrentLearning(model)

learning.learn(teach_cases, teach_answers, cal_count=1000, step=0.005)
learning.learn(teach_cases, teach_answers, cal_count=1000, step=0.0005)

# Show a preview
for i in range(3):
    print(f"Case {i+1}:")
    # print("  Inputs:", teach_cases[i])
    print("  Answer:", teach_answers[i])
    g=model.run(teach_cases[i][0])
    next(g)
    for j in teach_cases[i][1:]:
        # print(j)
        g.send(j)
    g.close()
    print("  Prediction:", model.output())