import re
import matplotlib.pyplot as plt

seed = 1
def extract_bleu_scores(file_path):
    bleu_scores = []
    
    # Regular expression to match the line containing validation bleu
    pattern = r'Validation loss: [\d.]+, validation bleu: ([\d.]+)'
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                bleu_scores.append(float(match.group(1)))
    
    return bleu_scores

def plot_bleu_scores(bleu_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(bleu_scores, marker='o', linestyle='-', color='b')
    plt.title('Validation BLEU Scores Over Time')
    plt.xlabel('Epoch/Iteration')
    plt.ylabel('Validation BLEU Score')
    plt.grid()
    # plt.xticks(range(len(bleu_scores)))  # Set x-ticks to match the number of scores
    plt.savefig(f'seed{seed}.png')

if __name__ == "__main__":
    # Replace 'your_log_file.txt' with the path to your log file
    file_path = f'mt2_3_2_{seed}.out'
    
    bleu_scores = extract_bleu_scores(file_path)
    
    if bleu_scores:
        print("Extracted Validation BLEU Scores:", bleu_scores)
        plot_bleu_scores(bleu_scores)
    else:
        print("No validation BLEU scores found in the file.")
