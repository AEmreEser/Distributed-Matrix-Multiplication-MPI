import re
import sys
from prettytable import PrettyTable

# Function to parse the log file and calculate averages
def parse_log(file_path):
    results = {}
    block_size = None

    # Read the file
    with open(file_path, 'r') as f:
        for line in f:
            # Match BLOCKSIZE lines
            block_match = re.match(r"BLOCKSIZE (\d+)", line)
            if block_match:
                block_size = int(block_match.group(1))
                results[block_size] = []
                continue

            # Match Execution time lines
            time_match = re.match(r"Execution time: ([0-9.]+)", line)
            if time_match and block_size is not None:
                results[block_size].append(float(time_match.group(1)))

    # Calculate averages for each block size
    averages = {block: sum(times) / len(times) if times else 0 for block, times in results.items()}
    return averages

# Function to generate a plain text table
def generate_plain_table(averages):
    table = PrettyTable()
    table.field_names = ["Block Size", "Average Execution Time"]
    for block, avg in sorted(averages.items()):
        table.add_row([block, f"{avg:.5f}"])
    return table

# Function to generate a LaTeX table
def generate_latex_table(averages):
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|c|c|}\n\\hline\n"
    latex_table += "Block Size & Average Execution Time \\\\\n\\hline\n"
    for block, avg in sorted(averages.items()):
        latex_table += f"{block} & {avg:.5f} \\\\\n"
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Execution Time Averages by Block Size}\n\\label{tab:exec_times}\n\\end{table}"
    return latex_table

# Function to generate a Markdown table
def generate_markdown_table(averages):
    markdown_table = "| Block Size | Average Execution Time |\n"
    markdown_table += "|------------|------------------------|\n"
    for block, avg in sorted(averages.items()):
        markdown_table += f"| {block}        | {avg:.5f}               |\n"
    return markdown_table

# Main function
def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_log.py <log_file> <output_format>")
        print("Output format options: plain, latex, markdown")
        sys.exit(1)

    file_path = sys.argv[1]
    output_format = sys.argv[2].lower()

    if output_format not in ["plain", "latex", "markdown"]:
        print("Invalid output format. Please choose 'plain', 'latex', or 'markdown'.")
        sys.exit(1)

    try:
        averages = parse_log(file_path)
        if output_format == "plain":
            table = generate_plain_table(averages)
            print(table)
        elif output_format == "latex":
            latex_table = generate_latex_table(averages)
            print(latex_table)
        elif output_format == "markdown":
            markdown_table = generate_markdown_table(averages)
            print(markdown_table)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

